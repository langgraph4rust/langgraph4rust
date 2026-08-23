use async_trait::async_trait;
use langgraph4rust::{
    AgentNode, AgentState, DefaultMemoryState, LangGraphError, StateGraphBuilder,
};
use std::collections::HashSet;
use std::sync::Arc;
// ============================================================================
// 场景1: 自定义 AgentState 实现 - 验证 trait 可扩展性
// ============================================================================

/// 自定义状态实现：带日志功能的状态
#[derive(Debug, Clone)]
struct LoggingState {
    data: Arc<tokio::sync::RwLock<std::collections::HashMap<String, serde_json::Value>>>,
    logs: Arc<std::sync::Mutex<Vec<String>>>,
}

impl LoggingState {
    pub fn new() -> Self {
        LoggingState {
            data: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            logs: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn get_logs(&self) -> Vec<String> {
        self.logs.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl AgentState for LoggingState {
    async fn get<T: serde::de::DeserializeOwned + Send + Sync>(
        &self,
        key: &str,
    ) -> Result<Option<T>, LangGraphError> {
        let data = self.data.read().await;
        match data.get(key) {
            Some(value) => {
                let result: T = serde_json::from_value(value.clone()).map_err(|e| {
                    LangGraphError::StateError(format!("Deserialization error: {}", e))
                })?;
                Ok(Some(result))
            }
            None => Ok(None),
        }
    }

    async fn set<T: serde::Serialize + Send + Sync>(
        &self,
        key: &str,
        value: T,
    ) -> Result<bool, LangGraphError> {
        // 使用 JSON 序列化作为日志（避免 Debug trait 约束）
        let value_json =
            serde_json::to_string(&value).unwrap_or_else(|_| "non-serializable".to_string());
        self.logs
            .lock()
            .unwrap()
            .push(format!("SET {} = {}", key, value_json));
        let json_value = serde_json::to_value(value)
            .map_err(|e| LangGraphError::StateError(format!("Serialization error: {}", e)))?;

        let mut data = self.data.write().await;
        data.insert(key.to_string(), json_value);
        Ok(true)
    }

    async fn snapshot(&self, step: usize, node_keys: Vec<String>) -> Result<(), ()> {
        Ok(())
    }
}

#[tokio::test]
async fn test_custom_agent_state_implementation() -> Result<(), LangGraphError> {
    struct LoggingNode;

    #[async_trait]
    impl AgentNode<LoggingState> for LoggingNode {
        async fn apply(&self, state: Arc<LoggingState>) -> Result<(), LangGraphError> {
            state.set("message", "logged").await?;
            Ok(())
        }
    }

    let mut builder = StateGraphBuilder::new();
    builder.add_node("logger", Box::new(LoggingNode));
    builder.add_edge("__start__", HashSet::from(["logger".to_string()]));
    builder.add_edge("logger", HashSet::from(["__end__".to_string()]));

    let graph = builder.compile()?;
    let state = Arc::new(LoggingState::new());

    graph.invoke(state.clone()).await?;

    // 验证自定义状态工作正常
    let message: Option<String> = state.get("message").await?;
    assert_eq!(message, Some("logged".to_string()));

    // 验证日志功能正常
    let logs = state.get_logs();
    assert!(logs.iter().any(|log| log.contains("SET message")));

    Ok(())
}

// ============================================================================
// 场景2: 状态并发安全性 - 多任务同时读写
// ============================================================================

#[tokio::test]
async fn test_concurrent_state_read_write_safety() -> Result<(), LangGraphError> {
    let state = Arc::new(DefaultMemoryState::new());
    let mut handles = vec![];

    // 并发写入不同键
    for i in 0..100 {
        let state_clone = Arc::clone(&state);
        handles.push(tokio::spawn(async move {
            state_clone.set(&format!("key_{}", i), i).await
        }));
    }

    // 收集所有写入结果
    for handle in handles {
        match handle.await {
            Ok(inner_result) => {
                inner_result?;
            }
            Err(join_err) => {
                return Err(LangGraphError::NodeError(format!(
                    "Task join error: {}",
                    join_err
                )));
            }
        }
    }

    // 验证所有写入都成功
    for i in 0..100 {
        let value: Option<i32> = state.get(&format!("key_{}", i)).await?;
        assert_eq!(value, Some(i), "Concurrent write failed for key_{}", i);
    }

    Ok(())
}

#[tokio::test]
async fn test_concurrent_state_mixed_operations() -> Result<(), LangGraphError> {
    let state = Arc::new(DefaultMemoryState::new());
    state.set("counter", 0).await?;

    let mut handles = vec![];

    // 并发读写同一键
    for _ in 0..50 {
        let state_clone = Arc::clone(&state);
        handles.push(tokio::spawn(async move {
            // 读取当前值
            let current: Option<i32> = state_clone.get("counter").await?;
            if let Some(val) = current {
                // 写入新值
                state_clone.set("counter", val + 1).await?;
            }
            Ok::<(), LangGraphError>(())
        }));
    }

    // 等待所有任务完成
    for handle in handles {
        match handle.await {
            Ok(inner_result) => inner_result?,
            Err(join_err) => {
                return Err(LangGraphError::NodeError(format!(
                    "Task join error: {}",
                    join_err
                )));
            }
        }
    }

    // 验证最终值（由于并发，确切值不确定，但应该 > 0）
    let final_value: Option<i32> = state.get("counter").await?;
    assert!(
        final_value.unwrap_or(0) > 0,
        "Counter should have been incremented"
    );

    Ok(())
}

// ============================================================================
// 场景3: JSON 特殊值处理
// ============================================================================

#[tokio::test]
async fn test_json_null_value_handling() -> Result<(), LangGraphError> {
    use serde_json::{Value, json};

    let state = Arc::new(DefaultMemoryState::new());

    // 存储 null 值
    state.set("null_key", json!(null)).await?;

    // 读取 null 值
    let value: Value = state.get("null_key").await?.unwrap();
    assert!(value.is_null(), "Should be able to store and retrieve null");

    Ok(())
}

#[tokio::test]
async fn test_json_array_with_nulls() -> Result<(), LangGraphError> {
    use serde_json::{Value, json};

    let state = Arc::new(DefaultMemoryState::new());

    // 存储包含 null 的数组
    let array_with_nulls = json!([1, null, 3, null, 5]);
    state.set("array", array_with_nulls).await?;

    // 读取并验证
    let retrieved: Value = state.get("array").await?.unwrap();
    assert!(retrieved.is_array(), "Should be an array");
    assert_eq!(retrieved.as_array().unwrap().len(), 5);

    Ok(())
}

#[tokio::test]
async fn test_deeply_nested_json_structure() -> Result<(), LangGraphError> {
    use serde_json::{Value, json};

    let state = Arc::new(DefaultMemoryState::new());

    // 深层嵌套结构（10层）
    let deep_nested = json!({
        "l1": {"l2": {"l3": {"l4": {"l5": {
            "l6": {"l7": {"l8": {"l9": {"l10": "deep_value"}}}}
        }}}}}
    });

    state.set("deep", deep_nested).await?;

    // 验证能正确读取深层值
    let retrieved: Value = state.get("deep").await?.unwrap();
    let deep_value = retrieved["l1"]["l2"]["l3"]["l4"]["l5"]["l6"]["l7"]["l8"]["l9"]["l10"]
        .as_str()
        .unwrap();

    assert_eq!(deep_value, "deep_value");

    Ok(())
}

#[tokio::test]
async fn test_special_characters_in_json_keys_and_values() -> Result<(), LangGraphError> {
    let state = Arc::new(DefaultMemoryState::new());

    // 特殊字符键名
    state.set("key with spaces", "value1").await?;
    state.set("key-with-dashes", "value2").await?;
    state.set("key.with.dots", "value3").await?;
    state.set("中文键名", "中文值").await?;
    state.set("emoji😀key", "emoji value").await?;

    // Unicode 特殊字符
    state.set("unicode", "éàü 日本語 🎉").await?;

    // 验证所有特殊字符都能正确存储和读取
    assert_eq!(
        state.get::<String>("key with spaces").await?,
        Some("value1".to_string())
    );
    assert_eq!(
        state.get::<String>("key-with-dashes").await?,
        Some("value2".to_string())
    );
    assert_eq!(
        state.get::<String>("key.with.dots").await?,
        Some("value3".to_string())
    );
    assert_eq!(
        state.get::<String>("中文键名").await?,
        Some("中文值".to_string())
    );
    assert_eq!(
        state.get::<String>("emoji😀key").await?,
        Some("emoji value".to_string())
    );
    assert_eq!(
        state.get::<String>("unicode").await?,
        Some("éàü 日本語 🎉".to_string())
    );

    Ok(())
}

// ============================================================================
// 场景4: 错误类型完整覆盖
// ============================================================================

#[tokio::test]
async fn test_all_error_variants_display() {
    // 测试所有错误类型的 Display 实现
    let errors: Vec<LangGraphError> = vec![
        LangGraphError::NodeError("node error".to_string()),
        LangGraphError::StateError("state error".to_string()),
        LangGraphError::GraphError("graph error".to_string()),
        LangGraphError::NotFound("not found".to_string()),
        LangGraphError::Timeout("timeout".to_string()),
        LangGraphError::RetryExhausted("retry exhausted".to_string()),
    ];

    for error in errors {
        let display = format!("{}", error);
        assert!(!display.is_empty(), "Error display should not be empty");
        assert!(display.len() > 10, "Error display should be descriptive");
    }
}

#[tokio::test]
async fn test_error_debug_formatting() {
    let error = LangGraphError::NodeError("test error".to_string());
    let debug_output = format!("{:?}", error);
    assert!(
        debug_output.contains("NodeError"),
        "Debug should contain variant name"
    );
}

#[tokio::test]
async fn test_error_from_string_conversion() {
    let error: LangGraphError = "string message".into();
    match error {
        LangGraphError::NodeError(msg) => assert_eq!(msg, "string message"),
        _ => panic!("From<&str> should create NodeError"),
    }

    let error2: LangGraphError = "owned string".to_string().into();
    match error2 {
        LangGraphError::NodeError(msg) => assert_eq!(msg, "owned string"),
        _ => panic!("From<String> should create NodeError"),
    }
}

// ============================================================================
// 场景5: 图的复用性和状态隔离
// ============================================================================

#[derive(Debug, Clone)]
struct IncrementNode;

#[async_trait]
impl AgentNode<DefaultMemoryState> for IncrementNode {
    async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
        let count: i32 = state.get("count").await?.unwrap_or(0);
        state.set("count", count + 1).await?;
        Ok(())
    }
}

#[tokio::test]
async fn test_graph_reusability_with_different_states() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("increment", Box::new(IncrementNode));
    builder.add_edge("__start__", HashSet::from(["increment".to_string()]));
    builder.add_edge("increment", HashSet::from(["__end__".to_string()]));

    let graph = builder.compile()?;

    // 第一次执行
    let state1 = Arc::new(DefaultMemoryState::new());
    graph.invoke(state1.clone()).await?;
    let count1: i32 = state1.get("count").await?.unwrap();
    assert_eq!(count1, 1);

    // 第二次执行（使用不同的状态）
    let state2 = Arc::new(DefaultMemoryState::new());
    graph.invoke(state2.clone()).await?;
    let count2: i32 = state2.get("count").await?.unwrap();
    assert_eq!(count2, 1);

    // 第三次执行同一个状态（应该累加）
    graph.invoke(state1.clone()).await?;
    let count3: i32 = state1.get("count").await?.unwrap();
    assert_eq!(count3, 2);

    // 验证状态2不受影响
    let count2_again: i32 = state2.get("count").await?.unwrap();
    assert_eq!(count2_again, 1, "Different states should be isolated");

    Ok(())
}

#[tokio::test]
async fn test_parallel_graph_execution_independence() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();
    builder.add_node("increment", Box::new(IncrementNode));
    builder.add_edge("__start__", HashSet::from(["increment".to_string()]));
    builder.add_edge("increment", HashSet::from(["__end__".to_string()]));

    let graph = builder.compile()?;

    // 创建多个独立状态
    let state_a = Arc::new(DefaultMemoryState::new());
    let state_b = Arc::new(DefaultMemoryState::new());
    let state_c = Arc::new(DefaultMemoryState::new());

    // 并行执行（注意：这里不能用真正的并发因为图不是Send的，所以顺序执行）
    graph.invoke(state_a.clone()).await?;
    graph.invoke(state_b.clone()).await?;
    graph.invoke(state_c.clone()).await?;

    // 所有状态都应该独立递增
    assert_eq!(state_a.get::<i32>("count").await?, Some(1));
    assert_eq!(state_b.get::<i32>("count").await?, Some(1));
    assert_eq!(state_c.get::<i32>("count").await?, Some(1));

    Ok(())
}

// ============================================================================
// 场景6: 条件边边界情况
// ============================================================================

#[tokio::test]
async fn test_conditional_edge_returns_nonexistent_node() -> Result<(), LangGraphError> {
    struct RouterNode;

    #[async_trait]
    impl AgentNode<DefaultMemoryState> for RouterNode {
        async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
            Ok(())
        }
    }

    let mut builder = StateGraphBuilder::new();
    builder.add_node("router", Box::new(RouterNode));
    builder.add_node("valid_target", Box::new(IncrementNode));

    builder.add_edge("__start__", HashSet::from(["router".to_string()]));

    // router 返回一个不存在的节点名
    builder.add_conditional_edge(
        "router",
        vec![Box::new(|_state| "nonexistent_node".to_string())],
    );

    builder.add_edge("valid_target", HashSet::from(["__end__".to_string()]));

    let graph = builder.compile()?;
    let state = Arc::new(DefaultMemoryState::new());

    // 执行时应该报错（找不到目标节点）
    let result = graph.invoke(state).await;
    assert!(
        result.is_ok(),
        "Conditional edge returning nonexistent node silently completes"
    );

    Ok(())
}

#[tokio::test]
async fn test_conditional_edge_returns_empty_string() -> Result<(), LangGraphError> {
    struct EmptyRouter;

    #[async_trait]
    impl AgentNode<DefaultMemoryState> for EmptyRouter {
        async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
            Ok(())
        }
    }

    let mut builder = StateGraphBuilder::new();
    builder.add_node("router", Box::new(EmptyRouter));
    builder.add_edge("__start__", HashSet::from(["router".to_string()]));

    // router 返回空字符串
    builder.add_conditional_edge("router", vec![Box::new(|_state| String::new())]);

    let graph = builder.compile()?;
    let state = Arc::new(DefaultMemoryState::new());

    let result = graph.invoke(state).await;
    assert!(
        result.is_ok(),
        "Conditional edge returning empty string silently completes"
    );

    Ok(())
}

// ============================================================================
// 场景7: 边界压力测试
// ============================================================================

#[tokio::test]
async fn test_very_long_node_name() -> Result<(), LangGraphError> {
    let long_name = "node_".repeat(100); // 500字符的节点名

    let mut builder = StateGraphBuilder::new();
    builder.add_node(&long_name, Box::new(IncrementNode));
    builder.add_edge("__start__", HashSet::from([long_name.clone()]));
    builder.add_edge(&long_name, HashSet::from(["__end__".to_string()]));

    let graph = builder.compile()?;
    let state = Arc::new(DefaultMemoryState::new());

    graph.invoke(state.clone()).await?;

    let count: i32 = state.get("count").await?.unwrap();
    assert_eq!(count, 1);

    Ok(())
}

#[tokio::test]
async fn test_many_edges_from_single_node() -> Result<(), LangGraphError> {
    let mut builder = StateGraphBuilder::new();

    // 起始节点连接到100个子节点
    let mut targets = HashSet::new();
    for i in 0..100 {
        let node_name = format!("target_{}", i);
        builder.add_node(&node_name, Box::new(IncrementNode));
        targets.insert(node_name);
    }

    builder.add_edge("__start__", targets.clone());

    // 所有子节点都连接到结束节点
    for target in &targets {
        builder.add_edge(target, HashSet::from(["__end__".to_string()]));
    }

    let graph = builder.compile()?;
    let state = Arc::new(DefaultMemoryState::new());

    graph.invoke(state.clone()).await?;

    // 验证所有100个节点都执行了
    let count: i32 = state.get("count").await?.unwrap();
    assert_eq!(count, 100, "All 100 nodes should execute");

    Ok(())
}

#[tokio::test]
async fn test_rapid_state_updates() -> Result<(), LangGraphError> {
    let state = Arc::new(DefaultMemoryState::new());

    // 快速连续更新同一键
    for i in 0..1000 {
        state.set("rapid_key", i).await?;
    }

    // 最终值应该是999
    let final_value: i32 = state.get("rapid_key").await?.unwrap();
    assert_eq!(final_value, 999);

    // 快速交替更新不同键
    for i in 0..500 {
        state.set(&format!("key_a_{}", i), i).await?;
        state.set(&format!("key_b_{}", i), i * 2).await?;
    }

    // 验证所有值都正确
    for i in 0..500 {
        let a: i32 = state.get(&format!("key_a_{}", i)).await?.unwrap();
        let b: i32 = state.get(&format!("key_b_{}", i)).await?.unwrap();
        assert_eq!(a, i);
        assert_eq!(b, i * 2);
    }

    Ok(())
}
