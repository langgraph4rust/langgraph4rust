//! `state_graph_stream` 模块的全场景测试。
//!
//! 通过公开 API `StateGraph::stream()` 验证推送式事件流的所有行为：
//! 事件顺序、步骤编号、并行计时、条件路由、错误路径、max_steps 截断、接收方提前丢弃等。

use langgraph4rust::{
    AgentNode, AgentState, DefaultMemoryState, END_NODE, LangGraphError, START_NODE,
    StateGraphBuilder, StreamEvent, StreamExt,
};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use serde::Serialize;
// ─── 测试节点 ────────────────────────────────────────────────────────────────

/// 计数节点：每次执行将 state["count"] 加 1
#[derive(Debug, Clone)]
struct CounterNode;

#[langgraph4rust::async_trait]
impl AgentNode<DefaultMemoryState> for CounterNode {
    async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
        let count: i32 = state.get("count").await?.unwrap_or(0);
        state.set("count", count + 1).await?;
        Ok(())
    }
}

/// 失败节点：总是返回错误
#[derive(Debug, Clone)]
struct FailingNode;

#[langgraph4rust::async_trait]
impl AgentNode<DefaultMemoryState> for FailingNode {
    async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
        Err(LangGraphError::NodeError("intentional failure".into()))
    }
}

/// 慢节点：休眠指定毫秒后写入 state
#[derive(Debug, Clone)]
struct SlowNode {
    ms: u64,
}

#[langgraph4rust::async_trait]
impl AgentNode<DefaultMemoryState> for SlowNode {
    async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
        tokio::time::sleep(Duration::from_millis(self.ms)).await;
        state.set("slow_done", true).await?;
        Ok(())
    }
}

/// 路由节点：根据 state["route"] 决定下一跳（用于条件边）
#[derive(Debug, Clone)]
struct RouterNode;

#[langgraph4rust::async_trait]
impl AgentNode<DefaultMemoryState> for RouterNode {
    async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
        // 仅作为路由锚点，不做实际工作
        let _ = state.get::<i32>("route").await?;
        Ok(())
    }
}

// ─── 辅助函数 ────────────────────────────────────────────────────────────────

/// 收集流中所有事件（泛型：支持任意 AgentState 实现）
async fn collect_events<S>(
    graph: Arc<langgraph4rust::StateGraph<S>>,
    state: Arc<S>,
) -> Vec<StreamEvent<S>>
where
    S: AgentState + Send + Sync + 'static,
{
    let mut rx = graph.stream(state);
    let mut events = Vec::new();
    while let Some(event) = rx.next().await {
        events.push(event);
    }
    events
}

/// 从事件流末尾提取 WorkflowError 携带的错误（测试辅助）
fn extract_error(events: &[StreamEvent<DefaultMemoryState>]) -> &LangGraphError {
    match events.last() {
        Some(StreamEvent::WorkflowError { error, .. }) => error,
        _ => panic!("expected WorkflowError as last event"),
    }
}

// ─── 1. 线性工作流：事件完整序列 ─────────────────────────────────────────────

/// start → A → end
/// 预期事件序列：WorkflowStarted, StepStarted, NodeStarted, NodeFinished,
///              RoutingDecision, WorkflowFinished
#[tokio::test]
async fn test_linear_event_sequence() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(graph, state.clone()).await;

    // 首尾事件
    assert!(matches!(events.first(), Some(StreamEvent::WorkflowStarted)));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));

    // __start__ 虚拟节点占 step 1（仅 RoutingDecision），真实节点从 step 2 开始
    // 事件序列：WorkflowStarted, RoutingDecision(start→a), StepStarted(a),
    //          NodeStarted(a), NodeFinished(a), RoutingDecision(a→end), WorkflowFinished
    assert_eq!(events.len(), 7, "expected 7 events, got {}", events.len());
    assert!(matches!(
        &events[1],
        StreamEvent::RoutingDecision { step: 1, .. }
    ));
    assert!(matches!(&events[2], StreamEvent::StepStarted { step: 2, nodes } if nodes == &["a"]));
    assert!(matches!(&events[3], StreamEvent::NodeStarted { step: 2, name } if name == "a"));
    assert!(matches!(&events[4], StreamEvent::NodeFinished { step: 2, name, .. } if name == "a"));
    assert!(matches!(
        &events[5],
        StreamEvent::RoutingDecision { step: 2, .. }
    ));

    // 状态被更新
    let count: i32 = state.get("count").await?.unwrap_or(0);
    assert_eq!(count, 1);
    Ok(())
}

// ─── 2. 步骤编号一致性 ──────────────────────────────────────────────────────

/// start → A → B → end（两步）
/// 验证 StepStarted / NodeStarted / NodeFinished / RoutingDecision 的 step 字段同步递增
#[tokio::test]
async fn test_step_index_consistency() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // __start__ 占 step 1（仅 RoutingDecision），真实节点 A 在 step 2，B 在 step 3
    let mut step1_events = 0usize;
    let mut step2_events = 0usize;
    let mut step3_events = 0usize;
    for e in &events {
        match e {
            StreamEvent::StepStarted { step, .. }
            | StreamEvent::NodeStarted { step, .. }
            | StreamEvent::NodeFinished { step, .. }
            | StreamEvent::RoutingDecision { step, .. } => match *step {
                1 => step1_events += 1,
                2 => step2_events += 1,
                3 => step3_events += 1,
                _ => panic!("unexpected step index: {}", step),
            },
            _ => {}
        }
    }
    // step1: 仅 RoutingDecision(__start__→a) = 1
    assert_eq!(
        step1_events, 1,
        "step 1 should have 1 event (RoutingDecision)"
    );
    // step2: StepStarted + NodeStarted + NodeFinished + RoutingDecision = 4
    assert_eq!(step2_events, 4, "step 2 should have 4 events");
    // step3: 同上 = 4
    assert_eq!(step3_events, 4, "step 3 should have 4 events");
    Ok(())
}

// ─── 3. WorkflowFinished 元数据 ─────────────────────────────────────────────

#[tokio::test]
async fn test_workflow_finished_metadata() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_node("c", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from(["c".to_string()]));
    builder.add_edge("c", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    if let Some(StreamEvent::WorkflowFinished {
        state: final_state,
        total_steps,
        elapsed,
    }) = events.last()
    {
        // total_steps 包含 __start__ 步和 __end__ 检测步
        assert_eq!(
            *total_steps, 5,
            "should execute 5 steps (start + 3 nodes + end)"
        );
        assert!(*elapsed > Duration::ZERO, "elapsed should be positive");
        // final_state 与传入的 state 是同一个 Arc
        let count: i32 = final_state.get("count").await?.unwrap_or(0);
        assert_eq!(count, 3);
    } else {
        panic!("last event should be WorkflowFinished");
    }
    Ok(())
}

// ─── 4. 并行节点：事件交错 & 独立计时 ───────────────────────────────────────

/// start → {slow_a, slow_b} → end
/// 两节点各休眠 50ms，并行执行时总耗时应 < 100ms（串行则 >= 100ms）
#[tokio::test]
async fn test_parallel_nodes_concurrent_timing() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("slow_a", Box::new(SlowNode { ms: 50 }));
    builder.add_node("slow_b", Box::new(SlowNode { ms: 50 }));
    builder.add_edge(
        START_NODE,
        HashSet::from(["slow_a".to_string(), "slow_b".to_string()]),
    );
    builder.add_edge("slow_a", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("slow_b", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // 收集两个节点的 elapsed
    let elapsed_values: Vec<Duration> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::NodeFinished { elapsed, .. } => Some(*elapsed),
            _ => None,
        })
        .collect();
    assert_eq!(elapsed_values.len(), 2, "should have 2 NodeFinished events");

    // 每个节点的 elapsed 应 >= 50ms（自身真实耗时）
    for d in &elapsed_values {
        assert!(
            *d >= Duration::from_millis(45),
            "node elapsed {:?} should be ~50ms",
            d
        );
    }

    // WorkflowFinished.elapsed 应 < 150ms（并行，非串行 100ms+）
    if let Some(StreamEvent::WorkflowFinished { elapsed, .. }) = events.last() {
        assert!(
            *elapsed < Duration::from_millis(150),
            "total elapsed {:?} suggests sequential execution",
            elapsed
        );
    }
    Ok(())
}

/// 验证并行步骤的 StepStarted.nodes 包含所有并行节点名
#[tokio::test]
async fn test_parallel_step_started_contains_all_nodes() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("x", Box::new(CounterNode));
    builder.add_node("y", Box::new(CounterNode));
    builder.add_node("z", Box::new(CounterNode));
    builder.add_edge(
        START_NODE,
        HashSet::from(["x".to_string(), "y".to_string(), "z".to_string()]),
    );
    builder.add_edge("x", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("y", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("z", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    let step_started = events
        .iter()
        .find(|e| matches!(e, StreamEvent::StepStarted { .. }));
    if let Some(StreamEvent::StepStarted { step, nodes }) = step_started {
        assert_eq!(*step, 2, "real nodes start at step 2");
        assert_eq!(nodes.len(), 3);
        let set: HashSet<&String> = nodes.iter().collect();
        assert!(set.contains(&&"x".to_string()));
        assert!(set.contains(&&"y".to_string()));
        assert!(set.contains(&&"z".to_string()));
    } else {
        panic!("should have StepStarted event");
    }
    Ok(())
}

// ─── 5. 条件路由 ────────────────────────────────────────────────────────────

/// start → router → (条件边根据 state["route"] 选 "left" 或 "right") → end
#[tokio::test]
async fn test_conditional_routing_decision() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("router", Box::new(RouterNode));
    builder.add_node("left", Box::new(CounterNode));
    builder.add_node("right", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["router".to_string()]));
    builder.add_conditional_edge(
        "router",
        vec![Box::new(|state: &DefaultMemoryState| {
            // 同步上下文中无法 await，用 blocking 方式读取（测试简化）
            // 这里直接返回固定值模拟路由
            let _ = state;
            "left".to_string()
        })],
    );
    builder.add_edge("left", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("right", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // 应存在 RoutingDecision 且 to_nodes 包含 "left"（从 router 出发的条件路由）
    let routing = events.iter().find(|e| matches!(
        e,
        StreamEvent::RoutingDecision { from_nodes, .. } if from_nodes.contains(&"router".to_string())
    ));
    assert!(routing.is_some(), "should have RoutingDecision from router");
    if let Some(StreamEvent::RoutingDecision {
        from_nodes,
        to_nodes,
        ..
    }) = routing
    {
        assert!(from_nodes.contains(&"router".to_string()));
        assert!(to_nodes.contains(&"left".to_string()));
    }

    // 最终成功
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    Ok(())
}

// ─── 6. 节点执行失败 → WorkflowError ────────────────────────────────────────

#[tokio::test]
async fn test_node_failure_emits_workflow_error() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("ok", Box::new(CounterNode));
    builder.add_node("fail", Box::new(FailingNode));
    builder.add_edge(START_NODE, HashSet::from(["ok".to_string()]));
    builder.add_edge("ok", HashSet::from(["fail".to_string()]));
    builder.add_edge("fail", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // 首事件 WorkflowStarted
    assert!(matches!(events.first(), Some(StreamEvent::WorkflowStarted)));
    // 末事件 WorkflowError（不是 WorkflowFinished）
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowError { .. })
    ));
    // 不应出现 WorkflowFinished
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, StreamEvent::WorkflowFinished { .. }))
    );

    // 错误内容验证（__start__=step1, ok=step2, fail=step3）
    if let Some(StreamEvent::WorkflowError { step, error, .. }) = events.last() {
        assert_eq!(*step, 3, "error should occur at step 3");
        assert!(matches!(error, LangGraphError::NodeError(_)));
    }
    Ok(())
}

/// 并行节点中一个失败 → WorkflowError，且 NodeFinished 仍被发射（失败节点也有）
#[tokio::test]
async fn test_parallel_node_failure() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("good", Box::new(CounterNode));
    builder.add_node("bad", Box::new(FailingNode));
    builder.add_edge(
        START_NODE,
        HashSet::from(["good".to_string(), "bad".to_string()]),
    );
    builder.add_edge("good", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("bad", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // 应以 WorkflowError 结束
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowError { .. })
    ));
    // 两个节点都应有 NodeStarted（并行启动）
    let started_count = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::NodeStarted { .. }))
        .count();
    assert_eq!(started_count, 2, "both nodes should start");
    Ok(())
}

// ─── 7. max_steps 耗尽 → WorkflowError ──────────────────────────────────────

/// 构造一个环：A → B → A → B → ...，设 max_steps=4，永远到不了 end
#[tokio::test]
async fn test_max_steps_exhaustion_emits_error() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.set_max_steps(4);
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from(["a".to_string()])); // 环！永远不到 end
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    // 末事件应为 WorkflowError（GraphError: Reached max_steps）
    if let Some(StreamEvent::WorkflowError { step, error, .. }) = events.last() {
        assert_eq!(*step, 4, "should stop at max_steps");
        let msg = error.to_string();
        assert!(
            msg.contains("max_steps"),
            "error should mention max_steps, got: {}",
            msg
        );
    } else {
        panic!("last event should be WorkflowError");
    }

    // 不应出现 WorkflowFinished
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, StreamEvent::WorkflowFinished { .. }))
    );

    // 状态仍被部分更新（执行了若干步）
    let count: i32 = state.get("count").await?.unwrap_or(0);
    assert!(count > 0, "some steps should have executed");
    Ok(())
}

// ─── 8. 接收方提前 drop → 驱动静默终止 ──────────────────────────────────────

#[tokio::test]
async fn test_receiver_drop_terminates_driver() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    for name in ["n1", "n2", "n3", "n4", "n5"] {
        builder.add_node(name, Box::new(SlowNode { ms: 20 }));
    }
    builder.add_edge(START_NODE, HashSet::from(["n1".to_string()]));
    builder.add_edge("n1", HashSet::from(["n2".to_string()]));
    builder.add_edge("n2", HashSet::from(["n3".to_string()]));
    builder.add_edge("n3", HashSet::from(["n4".to_string()]));
    builder.add_edge("n4", HashSet::from(["n5".to_string()]));
    builder.add_edge("n5", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let mut rx = graph.stream(Arc::new(DefaultMemoryState::new()));

    // 只取第一个事件后立即 drop
    let first = rx.next().await;
    assert!(matches!(first, Some(StreamEvent::WorkflowStarted)));
    drop(rx);

    // 给后台任务一点时间感知 drop 并退出（不会 panic）
    tokio::time::sleep(Duration::from_millis(50)).await;
    // 到这里没有 panic 即为通过
    Ok(())
}

// ─── 9. 多步状态累积 ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_state_accumulates_across_steps() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    for name in ["s1", "s2", "s3", "s4"] {
        builder.add_node(name, Box::new(CounterNode));
    }
    builder.add_edge(START_NODE, HashSet::from(["s1".to_string()]));
    builder.add_edge("s1", HashSet::from(["s2".to_string()]));
    builder.add_edge("s2", HashSet::from(["s3".to_string()]));
    builder.add_edge("s3", HashSet::from(["s4".to_string()]));
    builder.add_edge("s4", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    let count: i32 = state.get("count").await?.unwrap_or(0);
    assert_eq!(count, 4, "4 nodes should increment count 4 times");
    Ok(())
}

// ─── 10. NodeFinished.elapsed 精确性 ────────────────────────────────────────

/// 单慢节点（80ms），验证 NodeFinished.elapsed >= 80ms
#[tokio::test]
async fn test_node_finished_elapsed_accuracy() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("slow", Box::new(SlowNode { ms: 80 }));
    builder.add_edge(START_NODE, HashSet::from(["slow".to_string()]));
    builder.add_edge("slow", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    let node_finished = events
        .iter()
        .find(|e| matches!(e, StreamEvent::NodeFinished { name, .. } if name == "slow"));
    if let Some(StreamEvent::NodeFinished { elapsed, .. }) = node_finished {
        assert!(
            *elapsed >= Duration::from_millis(75),
            "elapsed {:?} should be >= ~80ms",
            elapsed
        );
    } else {
        panic!("should have NodeFinished for 'slow'");
    }
    Ok(())
}

// ─── 11. RoutingDecision 内容正确性 ─────────────────────────────────────────

#[tokio::test]
async fn test_routing_decision_from_to_nodes() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("alpha", Box::new(CounterNode));
    builder.add_node("beta", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["alpha".to_string()]));
    builder.add_edge("alpha", HashSet::from(["beta".to_string()]));
    builder.add_edge("beta", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // 共 3 条路由：__start__→alpha, alpha→beta, beta→__end__
    let routings: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::RoutingDecision { .. }))
        .collect();
    assert_eq!(routings.len(), 3, "should have 3 routing decisions");

    // 第一条：__start__ → alpha (step 1)
    if let StreamEvent::RoutingDecision {
        step,
        from_nodes,
        to_nodes,
    } = routings[0]
    {
        assert_eq!(*step, 1);
        assert!(from_nodes.contains(&"__start__".to_string()));
        assert!(to_nodes.contains(&"alpha".to_string()));
    }
    // 第二条：alpha → beta (step 2)
    if let StreamEvent::RoutingDecision {
        step,
        from_nodes,
        to_nodes,
    } = routings[1]
    {
        assert_eq!(*step, 2);
        assert!(from_nodes.contains(&"alpha".to_string()));
        assert!(to_nodes.contains(&"beta".to_string()));
    }
    // 第三条：beta → __end__ (step 3)
    if let StreamEvent::RoutingDecision {
        step,
        from_nodes,
        to_nodes,
    } = routings[2]
    {
        assert_eq!(*step, 3);
        assert!(from_nodes.contains(&"beta".to_string()));
        assert!(to_nodes.contains(&"__end__".to_string()));
    }
    Ok(())
}

// ─── 12. 事件总数精确验证 ───────────────────────────────────────────────────

/// N 步线性图的事件总数 = 1(WorkflowStarted) + 1(__start__ RoutingDecision) + N*(StepStarted+NodeStarted+NodeFinished+RoutingDecision) + 1(WorkflowFinished)
/// 即 3 + 4N
#[tokio::test]
async fn test_event_count_formula() -> Result<(), LangGraphError> {
    // N 个真实节点：step1=__start__(仅RoutingDecision), step2..N+1=真实节点(各4事件), stepN+2=__end__检测
    // 事件总数 = 1(WorkflowStarted) + 1(__start__ RoutingDecision) + N*4 + 1(WorkflowFinished) = 3 + 4N
    for n in 1..=6 {
        let mut builder = StateGraphBuilder::new();
        let names: Vec<String> = (0..n).map(|i| format!("node_{}", i)).collect();
        for name in &names {
            builder.add_node(name, Box::new(CounterNode));
        }
        builder.add_edge(START_NODE, HashSet::from([names[0].clone()]));
        for i in 0..n - 1 {
            builder.add_edge(&names[i], HashSet::from([names[i + 1].clone()]));
        }
        builder.add_edge(&names[n - 1], HashSet::from([END_NODE.to_string()]));
        let graph = Arc::new(builder.compile()?);

        let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
        let expected = 3 + 4 * n;
        assert_eq!(
            events.len(),
            expected,
            "n={}: expected {} events, got {}",
            n,
            expected,
            events.len()
        );
    }
    Ok(())
}

// ─── 13. WorkflowStarted 始终是第一个事件 ────────────────────────────────────

#[tokio::test]
async fn test_workflow_started_always_first() -> Result<(), LangGraphError> {
    // 成功路径
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);
    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    assert!(matches!(events[0], StreamEvent::WorkflowStarted));

    // 失败路径
    let mut builder = StateGraphBuilder::new();
    builder.add_node("f", Box::new(FailingNode));
    builder.add_edge(START_NODE, HashSet::from(["f".to_string()]));
    builder.add_edge("f", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);
    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    assert!(matches!(events[0], StreamEvent::WorkflowStarted));
    Ok(())
}

// ─── 14. WorkflowError 是失败时的最后一个事件（流随即关闭）────────────────────

#[tokio::test]
async fn test_workflow_error_is_terminal() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("fail", Box::new(FailingNode));
    builder.add_node("unreachable", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["fail".to_string()]));
    builder.add_edge("fail", HashSet::from(["unreachable".to_string()]));
    builder.add_edge("unreachable", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // WorkflowError 是最后一个事件
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowError { .. })
    ));
    // "unreachable" 节点不应有任何事件
    assert!(!events.iter().any(|e| matches!(
        e,
        StreamEvent::NodeStarted { name, .. } if name == "unreachable"
    )));
    Ok(())
}

// ─── 15. 菱形拓扑：fan-out 后 fan-in ────────────────────────────────────────

/// start → {a, b} → merge → end：并行分支汇聚到单一节点
#[tokio::test]
async fn test_diamond_fan_out_fan_in() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_node("merge", Box::new(CounterNode));
    builder.add_edge(
        START_NODE,
        HashSet::from(["a".to_string(), "b".to_string()]),
    );
    builder.add_edge("a", HashSet::from(["merge".to_string()]));
    builder.add_edge("b", HashSet::from(["merge".to_string()]));
    builder.add_edge("merge", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    // a, b, merge 各执行一次
    let count: i32 = state.get("count").await?.unwrap_or(0);
    assert_eq!(count, 3, "a + b + merge should run 3 times");
    // merge 只应启动一次（fan-in 去重）
    let merge_starts = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::NodeStarted { name, .. } if name == "merge"))
        .count();
    assert_eq!(merge_starts, 1, "merge should run exactly once");
    Ok(())
}

// ─── 16. 并行节点全部成功 ────────────────────────────────────────────────────

#[tokio::test]
async fn test_parallel_all_nodes_finish_successfully() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_node("c", Box::new(CounterNode));
    builder.add_edge(
        START_NODE,
        HashSet::from(["a".to_string(), "b".to_string(), "c".to_string()]),
    );
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("b", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("c", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    let finished = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::NodeFinished { .. }))
        .count();
    assert_eq!(finished, 3, "all 3 nodes should finish");
    let count: i32 = state.get("count").await?.unwrap_or(0);
    assert_eq!(count, 3);
    Ok(())
}

// ─── 17. WorkflowError 携带状态快照（部分更新保留）─────────────────────

#[tokio::test]
async fn test_workflow_error_carries_state_snapshot() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("ok", Box::new(CounterNode));
    builder.add_node("fail", Box::new(FailingNode));
    builder.add_edge(START_NODE, HashSet::from(["ok".to_string()]));
    builder.add_edge("ok", HashSet::from(["fail".to_string()]));
    builder.add_edge("fail", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    if let Some(StreamEvent::WorkflowError { state, .. }) = events.last() {
        // ok 节点已将 count 置 1，错误快照应保留该部分更新
        let count: i32 = state.get("count").await?.unwrap_or(0);
        assert_eq!(count, 1, "error snapshot should keep partial update");
    } else {
        panic!("last event should be WorkflowError");
    }
    Ok(())
}

// ─── 18. WorkflowFinished.state 与输入是同一个 Arc ─────────────────────────

#[tokio::test]
async fn test_workflow_finished_state_is_same_arc() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(graph, Arc::clone(&state)).await;

    if let Some(StreamEvent::WorkflowFinished {
        state: final_state, ..
    }) = events.last()
    {
        assert!(
            Arc::ptr_eq(&state, final_state),
            "finished state should be the same Arc as input"
        );
    } else {
        panic!("last event should be WorkflowFinished");
    }
    Ok(())
}

// ─── 19. 图复用：同一编译图顺序多次 stream ────────────────────────────────

#[tokio::test]
async fn test_graph_reuse_sequential_streams() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    // 两次独立执行，状态互不干扰
    for _ in 0..2 {
        let state = Arc::new(DefaultMemoryState::new());
        let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;
        assert!(matches!(
            events.last(),
            Some(StreamEvent::WorkflowFinished { .. })
        ));
        let count: i32 = state.get("count").await?.unwrap_or(0);
        assert_eq!(count, 1, "each run starts from fresh state");
    }
    Ok(())
}

// ─── 20. 并发流：同一 Arc 图同时跑多个 stream ──────────────────────────────

#[tokio::test]
async fn test_concurrent_streams_same_graph() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let (e1, e2, e3) = tokio::join!(
        collect_events(Arc::clone(&graph), Arc::new(DefaultMemoryState::new())),
        collect_events(Arc::clone(&graph), Arc::new(DefaultMemoryState::new())),
        collect_events(Arc::clone(&graph), Arc::new(DefaultMemoryState::new())),
    );
    for events in [&e1, &e2, &e3] {
        assert!(matches!(
            events.last(),
            Some(StreamEvent::WorkflowFinished { .. })
        ));
    }
    Ok(())
}

// ─── 21. 多 router 条件边：下一跳为各 router 结果并集 ─────────────────────

/// 单一条件边含两个 router（分别返回 t1 / t2），下一步应同时执行 t1 和 t2
#[tokio::test]
async fn test_multiple_conditional_routers_merge() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("split", Box::new(CounterNode));
    builder.add_node("t1", Box::new(CounterNode));
    builder.add_node("t2", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["split".to_string()]));
    builder.add_conditional_edge(
        "split",
        vec![
            Box::new(|_state: &DefaultMemoryState| "t1".to_string()),
            Box::new(|_state: &DefaultMemoryState| "t2".to_string()),
        ],
    );
    builder.add_edge("t1", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("t2", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    // 从 split 出发的 RoutingDecision 应同时包含 t1 和 t2
    let routing = events.iter().find(|e| matches!(
        e,
        StreamEvent::RoutingDecision { from_nodes, .. } if from_nodes.contains(&"split".to_string())
    ));
    if let Some(StreamEvent::RoutingDecision { to_nodes, .. }) = routing {
        assert!(to_nodes.contains(&"t1".to_string()), "missing t1");
        assert!(to_nodes.contains(&"t2".to_string()), "missing t2");
    } else {
        panic!("should have RoutingDecision from split");
    }
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    // t1, t2 各执行一次（split 也执行）→ count = 3
    let count: i32 = state.get("count").await?.unwrap_or(0);
    assert_eq!(count, 3);
    Ok(())
}

/// 补充：验证器禁止同一节点同时拥有静态边与条件边（compile 报错）
#[test]
fn test_node_cannot_have_both_static_and_conditional_edges() {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("split", Box::new(CounterNode));
    builder.add_node("t", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["split".to_string()]));
    builder.add_edge("split", HashSet::from(["t".to_string()]));
    builder.add_conditional_edge(
        "split",
        vec![Box::new(|_state: &DefaultMemoryState| "t".to_string())],
    );
    builder.add_edge("t", HashSet::from([END_NODE.to_string()]));

    let result = builder.compile();
    assert!(
        result.is_err(),
        "compile should reject mixed static + conditional edges"
    );
    let err = result.err().unwrap();
    assert!(matches!(err, LangGraphError::GraphError(_)));
    assert!(
        err.to_string()
            .contains("both static edges and conditional edges")
    );
}

// ─── 22. 条件边直接路由到 END ───────────────────────────────────────────────

#[tokio::test]
async fn test_conditional_edge_directly_to_end() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("router", Box::new(RouterNode));
    builder.add_edge(START_NODE, HashSet::from(["router".to_string()]));
    builder.add_conditional_edge(
        "router",
        vec![Box::new(|_state: &DefaultMemoryState| END_NODE.to_string())],
    );
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    // router 后直接结束，不应有其他节点启动
    let started = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::NodeStarted { .. }))
        .count();
    assert_eq!(started, 1, "only router should start");
    Ok(())
}

// ─── 23. 大扇出：触发有界通道背压 ───────────────────────────────────────────

/// 20 个并行节点，单步事件数(StepStarted+20*Started+20*Finished+Routing=42)超过通道容量32，
/// 验证背压下不死锁、全部完成
#[tokio::test]
async fn test_large_fan_out_backpressure() -> Result<(), LangGraphError> {
    const N: usize = 20;
    let mut builder = StateGraphBuilder::new();
    let names: Vec<String> = (0..N).map(|i| format!("p{}", i)).collect();
    for name in &names {
        builder.add_node(name, Box::new(CounterNode));
    }
    builder.add_edge(START_NODE, names.iter().cloned().collect());
    for name in &names {
        builder.add_edge(name, HashSet::from([END_NODE.to_string()]));
    }
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    let finished = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::NodeFinished { .. }))
        .count();
    assert_eq!(finished, N, "all {} nodes should finish", N);
    let count: i32 = state.get("count").await?.unwrap_or(0);
    assert_eq!(count, N as i32);
    Ok(())
}

// ─── 24. 自循环 + max_steps ─────────────────────────────────────────────────

/// a → a（自环），永远到不了 end，max_steps=3 截断
#[tokio::test]
async fn test_self_loop_max_steps() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.set_max_steps(3);
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["a".to_string()])); // 自环
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    if let Some(StreamEvent::WorkflowError { error, .. }) = events.last() {
        assert!(
            error.to_string().contains("max_steps"),
            "should report max_steps"
        );
    } else {
        panic!("last event should be WorkflowError");
    }
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, StreamEvent::WorkflowFinished { .. }))
    );
    Ok(())
}

// ─── 25. 失败节点仍发射 NodeFinished ────────────────────────────────────────

/// run_node_with_events 在 apply 返回后无条件发 NodeFinished，故失败节点也有该事件
#[tokio::test]
async fn test_failing_node_still_emits_node_finished() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("fail", Box::new(FailingNode));
    builder.add_edge(START_NODE, HashSet::from(["fail".to_string()]));
    builder.add_edge("fail", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    assert!(events.iter().any(|e| matches!(
        e,
        StreamEvent::NodeStarted { name, .. } if name == "fail"
    )));
    assert!(
        events.iter().any(|e| matches!(
            e,
            StreamEvent::NodeFinished { name, .. } if name == "fail"
        )),
        "failing node should still emit NodeFinished"
    );
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowError { .. })
    ));
    Ok(())
}

// ─── 26. 状态依赖的条件路由（自定义同步可读状态）─────────────────────

/// 同步可读的状态实现：router 是同步 Fn，需能同步读取节点异步写入的路由值
#[derive(Clone)]
struct SyncRouteState {
    data: Arc<std::sync::Mutex<std::collections::HashMap<String, serde_json::Value>>>,
}

impl SyncRouteState {
    fn new() -> Self {
        Self {
            data: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }
    /// 同步读取字符串值（供同步 router 使用）
    fn get_sync(&self, key: &str) -> Option<String> {
        self.data
            .lock()
            .unwrap()
            .get(key)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
    }
}

#[langgraph4rust::async_trait]
impl AgentState for SyncRouteState {
    async fn get<T: serde::de::DeserializeOwned + Send + Sync>(
        &self,
        key: &str,
    ) -> Result<Option<T>, LangGraphError> {
        let guard = self.data.lock().unwrap();
        Ok(guard
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok()))
    }
    async fn set<T: serde::Serialize + Send + Sync>(
        &self,
        key: &str,
        value: T,
    ) -> Result<bool, LangGraphError> {
        let v =
            serde_json::to_value(value).map_err(|e| LangGraphError::StateError(e.to_string()))?;
        self.data.lock().unwrap().insert(key.to_string(), v);
        Ok(true)
    }

    async fn snapshot(
        &self,
        step: usize,
        node_keys: Vec<String>,
    ) -> Result<(), ()>{
        Ok(())
    }
}

/// 写入路由值的节点
#[derive(Clone)]
struct SetRouteNode {
    target: String,
}

#[langgraph4rust::async_trait]
impl AgentNode<SyncRouteState> for SetRouteNode {
    async fn apply(&self, state: Arc<SyncRouteState>) -> Result<(), LangGraphError> {
        state.set("route", self.target.clone()).await?;
        Ok(())
    }
}

/// 打标记节点：访问时在 state 写入 key=true
#[derive(Clone)]
struct MarkNode {
    key: String,
}

#[langgraph4rust::async_trait]
impl AgentNode<SyncRouteState> for MarkNode {
    async fn apply(&self, state: Arc<SyncRouteState>) -> Result<(), LangGraphError> {
        state.set(&self.key, true).await?;
        Ok(())
    }
}

/// decide 节点将 route 置为 "right"，条件边同步读取后路由到 right 分支
#[tokio::test]
async fn test_state_dependent_conditional_routing() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::<SyncRouteState>::new();
    builder.add_node(
        "decide",
        Box::new(SetRouteNode {
            target: "right".to_string(),
        }),
    );
    builder.add_node(
        "left",
        Box::new(MarkNode {
            key: "left_visited".to_string(),
        }),
    );
    builder.add_node(
        "right",
        Box::new(MarkNode {
            key: "right_visited".to_string(),
        }),
    );
    builder.add_edge(START_NODE, HashSet::from(["decide".to_string()]));
    builder.add_conditional_edge(
        "decide",
        vec![Box::new(|state: &SyncRouteState| {
            state
                .get_sync("route")
                .unwrap_or_else(|| "left".to_string())
        })],
    );
    builder.add_edge("left", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("right", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(SyncRouteState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    let right: Option<bool> = state.get("right_visited").await?;
    let left: Option<bool> = state.get("left_visited").await?;
    assert_eq!(right, Some(true), "right branch should be visited");
    assert_eq!(left, None, "left branch should NOT be visited");
    Ok(())
}

// ─── 补充辅助节点 ─────────────────────────────────────────────────────────────

/// 写入固定整数的节点
#[derive(Debug, Clone)]
struct SetIntNode {
    key: String,
    value: i32,
}

#[langgraph4rust::async_trait]
impl AgentNode<DefaultMemoryState> for SetIntNode {
    async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
        state.set(&self.key, self.value).await?;
        Ok(())
    }
}

/// 读取 read_key 并将其两倍写入 write_key 的节点（验证跨步数据依赖）
#[derive(Debug, Clone)]
struct DoubleNode {
    read_key: String,
    write_key: String,
}

#[langgraph4rust::async_trait]
impl AgentNode<DefaultMemoryState> for DoubleNode {
    async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
        let v: i32 = state.get(&self.read_key).await?.unwrap_or(0);
        state.set(&self.write_key, v * 2).await?;
        Ok(())
    }
}

// ─── 27. 事件不变量：NodeStarted 与 NodeFinished 严格配对 ───────────────────

#[tokio::test]
async fn test_node_started_finished_pairing() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // (step, name) 集合：NodeStarted 与 NodeFinished 应完全一致
    let started: HashSet<(usize, String)> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::NodeStarted { step, name } => Some((*step, name.clone())),
            _ => None,
        })
        .collect();
    let finished: HashSet<(usize, String)> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::NodeFinished { step, name, .. } => Some((*step, name.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        started, finished,
        "every NodeStarted must pair with a NodeFinished"
    );
    assert_eq!(started.len(), 2, "two real nodes should run");
    Ok(())
}

// ─── 28. 顺序不变量：NodeStarted 总在其 NodeFinished 之前 ───────────────────

#[tokio::test]
async fn test_node_started_precedes_its_finished() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("slow", Box::new(SlowNode { ms: 20 }));
    builder.add_edge(START_NODE, HashSet::from(["slow".to_string()]));
    builder.add_edge("slow", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    let started_pos = events
        .iter()
        .position(|e| matches!(e, StreamEvent::NodeStarted { name, .. } if name == "slow"));
    let finished_pos = events
        .iter()
        .position(|e| matches!(e, StreamEvent::NodeFinished { name, .. } if name == "slow"));
    match (started_pos, finished_pos) {
        (Some(s), Some(f)) => assert!(s < f, "NodeStarted must come before NodeFinished"),
        _ => panic!("both NodeStarted and NodeFinished should exist"),
    }
    Ok(())
}

// ─── 29. 顺序不变量：StepStarted 在同步 NodeStarted 之前 ────────────────────

#[tokio::test]
async fn test_step_started_precedes_node_started() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // 真实节点在 step 2：StepStarted{2} 应出现在 NodeStarted{2} 之前
    let step_pos = events
        .iter()
        .position(|e| matches!(e, StreamEvent::StepStarted { step: 2, .. }));
    let node_pos = events
        .iter()
        .position(|e| matches!(e, StreamEvent::NodeStarted { step: 2, .. }));
    match (step_pos, node_pos) {
        (Some(sp), Some(np)) => assert!(sp < np, "StepStarted must precede NodeStarted"),
        _ => panic!("both StepStarted and NodeStarted for step 2 should exist"),
    }
    Ok(())
}

// ─── 30. __start__ 虚拟步不产生 StepStarted ─────────────────────────────────

#[tokio::test]
async fn test_no_step_started_for_start_step() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // step 1 是 __start__ 虚拟步，只有 RoutingDecision，绝不应有 StepStarted
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, StreamEvent::StepStarted { step: 1, .. })),
        "__start__ step (1) must not emit StepStarted"
    );
    Ok(())
}

// ─── 31. max_steps 精确边界：刚好足够→成功，差一步→失败 ───────────────────

/// start→a→end 需 total_steps=3（start路由 + a执行 + end检测）
#[tokio::test]
async fn test_max_steps_exact_boundary() -> Result<(), LangGraphError> {
    // max_steps = 3：刚好足够 → WorkflowFinished
    let mut builder = StateGraphBuilder::new();
    builder.set_max_steps(3);
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);
    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    match events.last() {
        Some(StreamEvent::WorkflowFinished { total_steps, .. }) => {
            assert_eq!(*total_steps, 3, "should finish exactly at max_steps");
        }
        other => panic!(
            "expected WorkflowFinished, got {:?}",
            std::mem::discriminant(other.unwrap())
        ),
    }

    // max_steps = 2：差一步 → WorkflowError(max_steps)
    let mut builder = StateGraphBuilder::new();
    builder.set_max_steps(2);
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);
    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    assert!(
        matches!(events.last(), Some(StreamEvent::WorkflowError { .. })),
        "one step short of max_steps should error"
    );
    Ok(())
}

// ─── 32. 条件路由返回未注册节点 → 运行时 NotFound 错误 ─────────────────────

/// 验证器只检查条件边源头，router 返回值的合法性在运行时才暴露
#[tokio::test]
async fn test_conditional_router_unknown_target_errors() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("router", Box::new(RouterNode));
    builder.add_edge(START_NODE, HashSet::from(["router".to_string()]));
    builder.add_conditional_edge(
        "router",
        vec![Box::new(|_state: &DefaultMemoryState| "ghost".to_string())],
    );
    // 能编译通过（ghost 未被静态检查）
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    match events.last() {
        Some(StreamEvent::WorkflowError { error, .. }) => {
            assert!(
                matches!(error, LangGraphError::GraphError(_)),
                "unknown router target should yield GraphError"
            );
        }
        _ => panic!("expected WorkflowError for unknown router target"),
    }
    Ok(())
}

// ─── 33. 跨步数据依赖：后续节点可读前置节点写入的值 ───────────────────────

#[tokio::test]
async fn test_data_dependency_across_steps() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node(
        "init",
        Box::new(SetIntNode {
            key: "n".to_string(),
            value: 5,
        }),
    );
    builder.add_node(
        "double",
        Box::new(DoubleNode {
            read_key: "n".to_string(),
            write_key: "n2".to_string(),
        }),
    );
    builder.add_edge(START_NODE, HashSet::from(["init".to_string()]));
    builder.add_edge("init", HashSet::from(["double".to_string()]));
    builder.add_edge("double", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    // double 读到 init 写入的 5，写出 10
    let n2: i32 = state.get("n2").await?.unwrap_or(0);
    assert_eq!(n2, 10, "downstream node should see upstream write");
    Ok(())
}

// ─── 34. WorkflowFinished.elapsed 不小于最慢节点耗时 ────────────────────────

#[tokio::test]
async fn test_workflow_finished_elapsed_lower_bound() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("fast", Box::new(SlowNode { ms: 30 }));
    builder.add_node("slow", Box::new(SlowNode { ms: 60 }));
    builder.add_edge(
        START_NODE,
        HashSet::from(["fast".to_string(), "slow".to_string()]),
    );
    builder.add_edge("fast", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("slow", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    let max_node_elapsed = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::NodeFinished { elapsed, .. } => Some(*elapsed),
            _ => None,
        })
        .max()
        .unwrap_or(Duration::ZERO);

    if let Some(StreamEvent::WorkflowFinished { elapsed, .. }) = events.last() {
        assert!(
            *elapsed >= max_node_elapsed,
            "workflow elapsed {:?} must encompass slowest node {:?}",
            elapsed,
            max_node_elapsed
        );
    } else {
        panic!("last event should be WorkflowFinished");
    }
    Ok(())
}

// ─── 35. 终结事件唯一性：有且仅有一个且位于末尾 ─────────────────────────────

#[tokio::test]
async fn test_exactly_one_terminal_event() -> Result<(), LangGraphError> {
    let count_terminal = |events: &Vec<StreamEvent<DefaultMemoryState>>| {
        events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    StreamEvent::WorkflowFinished { .. } | StreamEvent::WorkflowError { .. }
                )
            })
            .count()
    };

    // 成功路径
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);
    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    assert_eq!(count_terminal(&events), 1, "success: exactly one terminal");
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));

    // 失败路径
    let mut builder = StateGraphBuilder::new();
    builder.add_node("f", Box::new(FailingNode));
    builder.add_edge(START_NODE, HashSet::from(["f".to_string()]));
    builder.add_edge("f", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);
    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    assert_eq!(count_terminal(&events), 1, "failure: exactly one terminal");
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowError { .. })
    ));
    Ok(())
}

// ─── 补充辅助节点（第二批）───────────────────────────────────────────────

/// 错误类别：用于验证不同错误变体透过流传递的保真度
#[derive(Debug, Clone, Copy)]
enum ErrKind {
    Timeout,
    State,
    Retry,
}

/// 可配置错误节点：返回指定变体的错误
#[derive(Debug, Clone)]
struct ErrorNode {
    kind: ErrKind,
}

#[langgraph4rust::async_trait]
impl AgentNode<DefaultMemoryState> for ErrorNode {
    async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
        Err(match self.kind {
            ErrKind::Timeout => LangGraphError::Timeout("timed out".into()),
            ErrKind::State => LangGraphError::StateError("state corrupted".into()),
            ErrKind::Retry => LangGraphError::RetryExhausted("retries exhausted".into()),
        })
    }
}

/// 一次写入多个键的节点
#[derive(Debug, Clone)]
struct MultiKeyNode;

#[langgraph4rust::async_trait]
impl AgentNode<DefaultMemoryState> for MultiKeyNode {
    async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
        state.set("k1", 1i32).await?;
        state.set("k2", 2i32).await?;
        state.set("k3", 3i32).await?;
        Ok(())
    }
}

// ─── 36. WorkflowStarted 有且仅发射一次 ────────────────────────────────────

#[tokio::test]
async fn test_workflow_started_emitted_exactly_once() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    let started_count = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::WorkflowStarted))
        .count();
    assert_eq!(
        started_count, 1,
        "WorkflowStarted must be emitted exactly once"
    );
    Ok(())
}

// ─── 37. 错误变体保真度：Timeout/StateError/RetryExhausted 原样传递 ───────────

#[tokio::test]
async fn test_error_variant_fidelity() -> Result<(), LangGraphError> {
    let run = |kind: ErrKind| async move {
        let mut builder = StateGraphBuilder::new();
        builder.add_node("boom", Box::new(ErrorNode { kind }));
        builder.add_edge(START_NODE, HashSet::from(["boom".to_string()]));
        builder.add_edge("boom", HashSet::from([END_NODE.to_string()]));
        let graph = Arc::new(builder.compile().unwrap());
        collect_events(graph, Arc::new(DefaultMemoryState::new())).await
    };

    let e_timeout = run(ErrKind::Timeout).await;
    assert!(matches!(
        extract_error(&e_timeout),
        LangGraphError::Timeout(_)
    ));

    let e_state = run(ErrKind::State).await;
    assert!(matches!(
        extract_error(&e_state),
        LangGraphError::StateError(_)
    ));

    let e_retry = run(ErrKind::Retry).await;
    assert!(matches!(
        extract_error(&e_retry),
        LangGraphError::RetryExhausted(_)
    ));
    Ok(())
}

// ─── 38. 并发流状态隔离：各自 state 互不污染 ────────────────────────────────

#[tokio::test]
async fn test_concurrent_streams_state_isolation() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let s1 = Arc::new(DefaultMemoryState::new());
    let s2 = Arc::new(DefaultMemoryState::new());
    let (e1, e2) = tokio::join!(
        collect_events(Arc::clone(&graph), Arc::clone(&s1)),
        collect_events(Arc::clone(&graph), Arc::clone(&s2)),
    );
    assert!(matches!(
        e1.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    assert!(matches!(
        e2.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));

    // 每个流独立计数为 1，而非共享后的 2 → 证明状态隔离
    let c1: i32 = s1.get("count").await?.unwrap_or(0);
    let c2: i32 = s2.get("count").await?.unwrap_or(0);
    assert_eq!(c1, 1, "stream 1 state must not be contaminated");
    assert_eq!(c2, 1, "stream 2 state must not be contaminated");
    Ok(())
}

// ─── 39. 流终止后重复轮询始终返回 None（幂等终止）───────────────────────

#[tokio::test]
async fn test_stream_termination_is_idempotent() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let mut rx = graph.stream(Arc::new(DefaultMemoryState::new()));
    // 排干所有事件
    while rx.next().await.is_some() {}
    // 终止后多次轮询均为 None
    assert!(rx.next().await.is_none());
    assert!(rx.next().await.is_none());
    assert!(rx.next().await.is_none());
    Ok(())
}

// ─── 40. 并行节点共享同一步骤编号 ──────────────────────────────────────────

#[tokio::test]
async fn test_parallel_nodes_share_step_index() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_edge(
        START_NODE,
        HashSet::from(["a".to_string(), "b".to_string()]),
    );
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("b", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    let started_steps: HashSet<usize> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::NodeStarted { step, .. } => Some(*step),
            _ => None,
        })
        .collect();
    let finished_steps: HashSet<usize> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::NodeFinished { step, .. } => Some(*step),
            _ => None,
        })
        .collect();
    // 两个并行节点同属 step 2
    assert_eq!(
        started_steps,
        HashSet::from([2]),
        "parallel nodes share one step"
    );
    assert_eq!(finished_steps, HashSet::from([2]));
    Ok(())
}

// ─── 41. 节点一次写入多个键均被持久化 ────────────────────────────────────────

#[tokio::test]
async fn test_node_writes_multiple_keys() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("multi", Box::new(MultiKeyNode));
    builder.add_edge(START_NODE, HashSet::from(["multi".to_string()]));
    builder.add_edge("multi", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));

    let k1: i32 = state.get("k1").await?.unwrap_or(0);
    let k2: i32 = state.get("k2").await?.unwrap_or(0);
    let k3: i32 = state.get("k3").await?.unwrap_or(0);
    assert_eq!((k1, k2, k3), (1, 2, 3), "all keys must be persisted");
    Ok(())
}

// ─── 42. 事件计数跨多次运行保持确定性 ────────────────────────────────────────

#[tokio::test]
async fn test_event_count_deterministic_across_runs() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_node("c", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from(["c".to_string()]));
    builder.add_edge("c", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    // N=3 线性图：事件总数 = 3 + 4*3 = 15
    let expected = 3 + 4 * 3;
    for _ in 0..5 {
        let events = collect_events(Arc::clone(&graph), Arc::new(DefaultMemoryState::new())).await;
        assert_eq!(events.len(), expected, "event count must be deterministic");
    }
    Ok(())
}

// ─── 43. 线性图 RoutingDecision 数量 = N + 1 ─────────────────────────────────

#[tokio::test]
async fn test_routing_decision_count_linear() -> Result<(), LangGraphError> {
    // start → a → b → end（N=2 真实节点）
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    let routing_count = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::RoutingDecision { .. }))
        .count();
    // __start__ 路由 + 每个真实节点路由 = N + 1 = 3
    assert_eq!(
        routing_count, 3,
        "linear N=2 graph should have N+1 routing decisions"
    );
    Ok(())
}

// ─── 44. 菱形图 total_steps：start→{a,b}→merge→end = 4 步 ───────────────────

#[tokio::test]
async fn test_diamond_total_steps() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_node("merge", Box::new(CounterNode));
    builder.add_edge(
        START_NODE,
        HashSet::from(["a".to_string(), "b".to_string()]),
    );
    builder.add_edge("a", HashSet::from(["merge".to_string()]));
    builder.add_edge("b", HashSet::from(["merge".to_string()]));
    builder.add_edge("merge", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    match events.last() {
        Some(StreamEvent::WorkflowFinished { total_steps, .. }) => {
            // step1=start路由, step2={a,b}并行, step3=merge, step4=end检测
            assert_eq!(*total_steps, 4, "diamond graph should take 4 steps");
        }
        _ => panic!("expected WorkflowFinished"),
    }
    Ok(())
}

// ─── 45. 慢消费者背压：逐事件延迟不丢失不乱序 ───────────────────────────

#[tokio::test]
async fn test_slow_consumer_no_event_loss() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_node("c", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from(["c".to_string()]));
    builder.add_edge("c", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let mut rx = graph.stream(Arc::new(DefaultMemoryState::new()));
    let mut events = Vec::new();
    while let Some(e) = rx.next().await {
        events.push(e);
        // 慢消费者：每次轮询后休眠，迫使生产者受背压阻塞
        tokio::time::sleep(Duration::from_millis(3)).await;
    }
    // N=3 线性图：3 + 4*3 = 15 个事件，一个不多一个不少
    assert_eq!(events.len(), 15, "slow consumer must not lose events");
    assert!(matches!(events.first(), Some(StreamEvent::WorkflowStarted)));
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    Ok(())
}

// ─── 46. 深线性链压力：50 节点 ───────────────────────────────────────────────

#[tokio::test]
async fn test_deep_linear_chain_stress() -> Result<(), LangGraphError> {
    const N: usize = 50;
    let mut builder = StateGraphBuilder::new();
    let names: Vec<String> = (0..N).map(|i| format!("n{i}")).collect();
    for name in &names {
        builder.add_node(name, Box::new(CounterNode));
    }
    builder.add_edge(START_NODE, HashSet::from([names[0].clone()]));
    for w in names.windows(2) {
        builder.add_edge(&w[0], HashSet::from([w[1].clone()]));
    }
    builder.add_edge(&names[N - 1], HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    assert_eq!(
        events.len(),
        3 + 4 * N,
        "event count formula must hold for deep chain"
    );
    if let Some(StreamEvent::WorkflowFinished { total_steps, .. }) = events.last() {
        assert_eq!(*total_steps, N + 2);
    } else {
        panic!("expected WorkflowFinished");
    }
    let count: i32 = state.get("count").await?.unwrap_or(0);
    assert_eq!(count, N as i32, "every node should have incremented count");
    Ok(())
}

// ─── 47. 并行节点写入不同键均可见 ────────────────────────────────────────────

#[tokio::test]
async fn test_parallel_nodes_write_distinct_keys() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node(
        "wx",
        Box::new(SetIntNode {
            key: "x".to_string(),
            value: 100,
        }),
    );
    builder.add_node(
        "wy",
        Box::new(SetIntNode {
            key: "y".to_string(),
            value: 200,
        }),
    );
    builder.add_edge(
        START_NODE,
        HashSet::from(["wx".to_string(), "wy".to_string()]),
    );
    builder.add_edge("wx", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("wy", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;
    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));

    let x: i32 = state.get("x").await?.unwrap_or(0);
    let y: i32 = state.get("y").await?.unwrap_or(0);
    assert_eq!((x, y), (100, 200), "both parallel writes must be visible");
    Ok(())
}

// ─── 48. 图复用：不同初始状态产生不同结果 ────────────────────────────────────

#[tokio::test]
async fn test_graph_reuse_with_different_initial_state() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    // 初始 count=10 → 执行后 11
    let s1 = Arc::new(DefaultMemoryState::new());
    s1.set("count", 10i32).await?;
    let _ = collect_events(Arc::clone(&graph), Arc::clone(&s1)).await;
    let c1: i32 = s1.get("count").await?.unwrap_or(0);
    assert_eq!(c1, 11);

    // 全新状态 → 1
    let s2 = Arc::new(DefaultMemoryState::new());
    let _ = collect_events(Arc::clone(&graph), Arc::clone(&s2)).await;
    let c2: i32 = s2.get("count").await?.unwrap_or(0);
    assert_eq!(c2, 1);
    Ok(())
}

// ─── 49. 并发流混合成败：同图上一成功一失败互不干扰 ───────────────────────

#[tokio::test]
async fn test_concurrent_streams_mixed_success_failure() -> Result<(), LangGraphError> {
    // 成功图
    let mut ok_builder = StateGraphBuilder::new();
    ok_builder.add_node("a", Box::new(CounterNode));
    ok_builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    ok_builder.add_edge("a", HashSet::from([END_NODE.to_string()]));
    let ok_graph = Arc::new(ok_builder.compile()?);

    // 失败图
    let mut fail_builder = StateGraphBuilder::new();
    fail_builder.add_node("f", Box::new(FailingNode));
    fail_builder.add_edge(START_NODE, HashSet::from(["f".to_string()]));
    fail_builder.add_edge("f", HashSet::from([END_NODE.to_string()]));
    let fail_graph = Arc::new(fail_builder.compile()?);

    let (ok_events, fail_events) = tokio::join!(
        collect_events(Arc::clone(&ok_graph), Arc::new(DefaultMemoryState::new())),
        collect_events(Arc::clone(&fail_graph), Arc::new(DefaultMemoryState::new())),
    );
    assert!(matches!(
        ok_events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    assert!(matches!(
        fail_events.last(),
        Some(StreamEvent::WorkflowError { .. })
    ));
    Ok(())
}

// ─── 50. 跨步顺序不变量：step N 的 RoutingDecision 先于 step N+1 的 StepStarted ───

#[tokio::test]
async fn test_cross_step_event_ordering() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["a".to_string()]));
    builder.add_edge("a", HashSet::from(["b".to_string()]));
    builder.add_edge("b", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // RoutingDecision{step:2} 必须出现在 StepStarted{step:3} 之前
    let route2 = events
        .iter()
        .position(|e| matches!(e, StreamEvent::RoutingDecision { step: 2, .. }));
    let step3 = events
        .iter()
        .position(|e| matches!(e, StreamEvent::StepStarted { step: 3, .. }));
    match (route2, step3) {
        (Some(r), Some(s)) => assert!(
            r < s,
            "routing of step N must precede StepStarted of step N+1"
        ),
        _ => panic!("both events should exist"),
    }
    Ok(())
}

// ─── 51. 多级条件路由链：连续两步条件决策 ────────────────────────────────────

#[tokio::test]
async fn test_conditional_chain_multiple_steps() -> Result<(), LangGraphError> {
    // start → r1 ─(cond)→ r2 ─(cond)→ end
    let mut builder = StateGraphBuilder::new();
    builder.add_node("r1", Box::new(RouterNode));
    builder.add_node("r2", Box::new(RouterNode));
    builder.add_edge(START_NODE, HashSet::from(["r1".to_string()]));
    builder.add_conditional_edge(
        "r1",
        vec![Box::new(|_s: &DefaultMemoryState| "r2".to_string())],
    );
    builder.add_conditional_edge(
        "r2",
        vec![Box::new(|_s: &DefaultMemoryState| END_NODE.to_string())],
    );
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    // r1 和 r2 都应被执行
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::NodeStarted { name, .. } if name == "r1"))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::NodeStarted { name, .. } if name == "r2"))
    );
    // 共 3 条 RoutingDecision：start→r1, r1→r2, r2→end
    let routing_count = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::RoutingDecision { .. }))
        .count();
    assert_eq!(routing_count, 3);
    Ok(())
}

// ─── 52. 并行执行中弃流：图不被污染，仍可复用 ────────────────────────────────

#[tokio::test]
async fn test_abandoned_stream_does_not_poison_graph() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("s1", Box::new(SlowNode { ms: 150 }));
    builder.add_node("s2", Box::new(SlowNode { ms: 150 }));
    builder.add_edge(
        START_NODE,
        HashSet::from(["s1".to_string(), "s2".to_string()]),
    );
    builder.add_edge("s1", HashSet::from([END_NODE.to_string()]));
    builder.add_edge("s2", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    // 在并行慢节点执行期间丢弃接收方
    let mut rx = Arc::clone(&graph).stream(Arc::new(DefaultMemoryState::new()));
    let _ = rx.next().await; // WorkflowStarted
    drop(rx);
    tokio::time::sleep(Duration::from_millis(30)).await;

    // 同一图仍可正常跑完一个新流
    let events = collect_events(Arc::clone(&graph), Arc::new(DefaultMemoryState::new())).await;
    assert!(
        matches!(events.last(), Some(StreamEvent::WorkflowFinished { .. })),
        "graph must remain usable after an abandoned stream"
    );
    Ok(())
}

// ─── 53. NodeFinished.elapsed 不小于节点自身休眠时长 ─────────────────────────

#[tokio::test]
async fn test_node_finished_elapsed_lower_bound() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("slow", Box::new(SlowNode { ms: 60 }));
    builder.add_edge(START_NODE, HashSet::from(["slow".to_string()]));
    builder.add_edge("slow", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    let elapsed = events.iter().find_map(|e| match e {
        StreamEvent::NodeFinished { name, elapsed, .. } if name == "slow" => Some(*elapsed),
        _ => None,
    });
    match elapsed {
        Some(d) => assert!(
            d >= Duration::from_millis(55),
            "elapsed {:?} should reflect the 60ms sleep",
            d
        ),
        None => panic!("NodeFinished for slow node should exist"),
    }
    Ok(())
}

// ─── 54. 首个真实节点即失败：WorkflowError.step == 2 ─────────────────────────

#[tokio::test]
async fn test_failure_at_first_real_node_step() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("f", Box::new(FailingNode));
    builder.add_edge(START_NODE, HashSet::from(["f".to_string()]));
    builder.add_edge("f", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;
    match events.last() {
        Some(StreamEvent::WorkflowError { step, .. }) => {
            // __start__ 占 step1，首个真实节点 f 在 step2 失败
            assert_eq!(*step, 2, "first real node fails at step 2");
        }
        _ => panic!("expected WorkflowError"),
    }
    Ok(())
}

// ─── 55. 多起始节点流式执行 ────────────────────────────────────────────────

/// 验证 add_start_node 添加的多个起始节点在流式执行中都能正确工作
#[tokio::test]
async fn test_stream_multi_start_nodes() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("a", Box::new(CounterNode));
    builder.add_node("b", Box::new(CounterNode));
    builder.add_node("merge", Box::new(CounterNode));
    builder.set_start_node("a");
    builder.add_start_node("b");
    builder.add_edge("a", HashSet::from(["merge".to_string()]));
    builder.add_edge("b", HashSet::from(["merge".to_string()]));
    builder.add_edge("merge", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    // a 和 b 并行执行，merge 执行一次
    let count: i32 = state.get("count").await?.unwrap_or(0);
    assert_eq!(count, 3, "a, b, merge should all execute");

    // 并行步应包含 a 和 b
    let step_started = events.iter().find(|e| {
        matches!(
            e,
            StreamEvent::StepStarted { nodes, .. } if nodes.len() == 2
        )
    });
    assert!(
        step_started.is_some(),
        "should have a step with 2 parallel nodes"
    );
    Ok(())
}

// ─── 56. 自定义结束节点流式执行 ────────────────────────────────────────────

/// 验证 set_end_node 自定义结束节点在流式执行中正常工作
#[tokio::test]
async fn test_stream_custom_end_node() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.set_end_node("finish");
    builder.add_node("step", Box::new(CounterNode));
    builder.add_edge(START_NODE, HashSet::from(["step".to_string()]));
    builder.add_edge("step", HashSet::from(["finish".to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(DefaultMemoryState::new());
    let events = collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    let count: i32 = state.get("count").await?.unwrap_or(0);
    assert_eq!(count, 1, "single step should execute");

    // 路由决策应包含自定义结束节点
    let routing = events.iter().find(|e| matches!(
        e,
        StreamEvent::RoutingDecision { to_nodes, .. } if to_nodes.contains(&"finish".to_string())
    ));
    assert!(
        routing.is_some(),
        "should have routing decision to custom end node 'finish'"
    );
    Ok(())
}

// ─── 57. StreamEvent Debug 格式化 ──────────────────────────────────────────

/// 简单的 Debug 状态实现，用于测试 StreamEvent 的 Debug 格式化
#[derive(Debug, Clone)]
struct DebugState;

#[langgraph4rust::async_trait]
impl AgentState for DebugState {
    async fn get<T: serde::de::DeserializeOwned + Send + Sync>(
        &self,
        _key: &str,
    ) -> Result<Option<T>, LangGraphError> {
        Ok(None)
    }
    async fn set<T: serde::Serialize + Send + Sync>(
        &self,
        _key: &str,
        _value: T,
    ) -> Result<bool, LangGraphError> {
        Ok(true)
    }

    async fn snapshot(
        &self,
        step: usize,
        node_keys: Vec<String>,
    ) -> Result<(), ()>{
        Ok(())
    }
}

/// 验证所有 StreamEvent 变体的 Debug 输出都包含有意义的信息
#[test]
fn test_stream_event_debug_format() {
    let events: Vec<StreamEvent<DebugState>> = vec![
        StreamEvent::WorkflowStarted,
        StreamEvent::StepStarted {
            step: 1,
            nodes: vec!["a".to_string()],
        },
        StreamEvent::NodeStarted {
            step: 1,
            name: "a".to_string(),
        },
        StreamEvent::NodeFinished {
            step: 1,
            name: "a".to_string(),
            elapsed: std::time::Duration::from_millis(10),
        },
        StreamEvent::RoutingDecision {
            step: 1,
            from_nodes: vec!["a".to_string()],
            to_nodes: vec!["b".to_string()],
        },
        StreamEvent::WorkflowFinished {
            state: Arc::new(DebugState),
            total_steps: 3,
            elapsed: std::time::Duration::from_millis(50),
        },
        StreamEvent::WorkflowError {
            state: Arc::new(DebugState),
            step: 2,
            error: LangGraphError::NodeError("test".into()),
        },
    ];

    for event in &events {
        let debug = format!("{:?}", event);
        assert!(
            !debug.is_empty(),
            "Debug output should not be empty for event variant"
        );
    }

    // 验证具体变体包含关键信息
    assert!(format!("{:?}", &events[0]).contains("WorkflowStarted"));
    assert!(format!("{:?}", &events[1]).contains("StepStarted"));
    assert!(format!("{:?}", &events[2]).contains("NodeStarted"));
    assert!(format!("{:?}", &events[3]).contains("NodeFinished"));
    assert!(format!("{:?}", &events[4]).contains("RoutingDecision"));
    assert!(format!("{:?}", &events[5]).contains("WorkflowFinished"));
    assert!(format!("{:?}", &events[6]).contains("WorkflowError"));
}

// ─── 58. run_driver 节点查找失败 ───────────────────────────────────────────

/// 验证当节点查找失败时（如条件边返回不存在的节点），流式执行产生 WorkflowError
#[tokio::test]
async fn test_stream_node_lookup_failure() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("router", Box::new(RouterNode));
    builder.add_edge(START_NODE, HashSet::from(["router".to_string()]));
    // 条件边返回一个不存在的节点，但 router 被执行后，下一轮 get_node_by_keys
    // 会尝试查找 ghost 节点，find 不到则 nodes 为空
    builder.add_conditional_edge(
        "router",
        vec![Box::new(|_state: &DefaultMemoryState| "ghost".to_string())],
    );
    let graph = Arc::new(builder.compile()?);

    let events = collect_events(graph, Arc::new(DefaultMemoryState::new())).await;

    // 应该收到 WorkflowError，因为 ghost 不是已注册节点，导致循环无法到达 end_node
    match events.last() {
        Some(StreamEvent::WorkflowError { error, .. }) => {
            let msg = error.to_string();
            assert!(
                msg.contains("Reached max_steps") || msg.contains("ghost"),
                "error should indicate graph issue, got: {}",
                msg
            );
        }
        other => {
            panic!("expected WorkflowError, got: {:?}", other.map(|_| ()));
        }
    }
    Ok(())
}

// ─── 59. 流式执行 + 自定义 AgentState ──────────────────────────────────────

/// 自定义状态实现：用于流式执行测试
#[derive(Debug, Clone)]
struct CounterState {
    data: Arc<std::sync::Mutex<std::collections::HashMap<String, serde_json::Value>>>,
}

impl CounterState {
    fn new() -> Self {
        Self {
            data: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        }
    }
}

#[langgraph4rust::async_trait]
impl AgentState for CounterState {
    async fn get<T: serde::de::DeserializeOwned + Send + Sync>(
        &self,
        key: &str,
    ) -> Result<Option<T>, LangGraphError> {
        let guard = self.data.lock().unwrap();
        Ok(guard
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok()))
    }
    async fn set<T: serde::Serialize + Send + Sync>(
        &self,
        key: &str,
        value: T,
    ) -> Result<bool, LangGraphError> {
        let v =
            serde_json::to_value(value).map_err(|e| LangGraphError::StateError(e.to_string()))?;
        self.data.lock().unwrap().insert(key.to_string(), v);
        Ok(true)
    }

    async fn snapshot(
        &self,
        step: usize,
        node_keys: Vec<String>,
    ) -> Result<(), ()>{
        Ok(())
    }
}

/// 自定义状态节点：写入固定值
#[derive(Debug, Clone)]
struct SetValueNode {
    key: String,
    value: i32,
}

#[langgraph4rust::async_trait]
impl AgentNode<CounterState> for SetValueNode {
    async fn apply(&self, state: Arc<CounterState>) -> Result<(), LangGraphError> {
        state.set(&self.key, self.value).await?;
        Ok(())
    }
}

/// 验证流式执行能使用自定义 AgentState 实现
#[tokio::test]
async fn test_stream_with_custom_agent_state() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::<CounterState>::new();
    builder.add_node(
        "init",
        Box::new(SetValueNode {
            key: "n".to_string(),
            value: 42,
        }),
    );
    builder.add_node(
        "double",
        Box::new(SetValueNode {
            key: "n2".to_string(),
            value: 84,
        }),
    );
    builder.add_edge(START_NODE, HashSet::from(["init".to_string()]));
    builder.add_edge("init", HashSet::from(["double".to_string()]));
    builder.add_edge("double", HashSet::from([END_NODE.to_string()]));
    let graph = Arc::new(builder.compile()?);

    let state = Arc::new(CounterState::new());
    let events: Vec<StreamEvent<CounterState>> =
        collect_events(Arc::clone(&graph), Arc::clone(&state)).await;

    assert!(matches!(
        events.last(),
        Some(StreamEvent::WorkflowFinished { .. })
    ));
    // 验证事件序列包含关键事件
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::WorkflowStarted))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::NodeStarted { name, .. } if name == "init"))
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, StreamEvent::NodeFinished { name, .. } if name == "init"))
    );

    // 验证自定义状态的数据被正确写入
    let n: i32 = state.get("n").await?.unwrap_or(0);
    let n2: i32 = state.get("n2").await?.unwrap_or(0);
    assert_eq!(n, 42, "init node should write 42");
    assert_eq!(n2, 84, "double node should write 84");
    Ok(())
}
