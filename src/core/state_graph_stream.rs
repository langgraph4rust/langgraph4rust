//! Real-time, push-based event streaming for graph execution.
//!
//! This module implements [`StateGraph::stream`], an alternative to
//! [`StateGraph::invoke`] that observes a workflow *while it runs* instead of
//! only receiving the final result. It mirrors the queue-based streaming model
//! of Python's LangGraph.
//!
//! # Architecture
//!
//! ```text
//!   tokio::spawn ──> run_driver ──> bounded mpsc channel ──> ReceiverStream ──> consumer
//!        (background task)            (capacity 32)          (Stream impl)
//! ```
//!
//! - [`StateGraph::stream`] spawns a **background driver task** (`run_driver`)
//!   and immediately returns a [`ReceiverStream`] wrapping the channel receiver.
//! - The driver pushes [`StreamEvent`]s into a **bounded** channel; when the
//!   consumer is slow, the driver awaits `send`, providing natural backpressure
//!   instead of unbounded buffering.
//! - Nodes within a step run as **concurrent futures** (via [`join_all`]); each
//!   emits its own [`StreamEvent::NodeStarted`] / [`StreamEvent::NodeFinished`]
//!   at the real moment it starts and finishes. Events therefore interleave in
//!   true execution order, and `elapsed` is each node's actual execution time.
//!
//! # Event lifecycle
//!
//! ```text
//! WorkflowStarted
//!   ├─ StepStarted { step, nodes }
//!   ├─ NodeStarted { step, name }      ┐ repeated per node,
//!   ├─ NodeFinished { step, name, .. } ┘ possibly interleaved
//!   ├─ RoutingDecision { step, from_nodes, to_nodes }
//!   └─ ... (next step) ...
//! WorkflowFinished { .. }   ── or ──   WorkflowError { .. }
//! ```
//!
//! The stream always terminates with exactly one of [`StreamEvent::WorkflowFinished`]
//! (success) or [`StreamEvent::WorkflowError`] (failure) as its final item.
//!
//! # Execution model & step indexing
//!
//! - The virtual [`__start__`](crate::START_NODE) node occupies **step 1** and emits
//!   only a [`StreamEvent::RoutingDecision`] (no `StepStarted` / node events);
//!   real nodes begin at **step 2**.
//! - For a linear graph with `N` real nodes (`start → N nodes → end`): total
//!   events = `3 + 4N` and [`WorkflowFinished.total_steps`](StreamEvent::WorkflowFinished) = `N + 2`
//!   (the extra two are the `__start__` routing step and the `__end__` detection step).
//! - If the step budget ([`max_steps`](crate::StateGraphBuilder::set_max_steps)) is
//!   exhausted before reaching the end node (e.g. a cycle), the stream emits a
//!   [`StreamEvent::WorkflowError`] wrapping [`LangGraphError::GraphError`] rather
//!   than reporting success.
//!
//! # Error semantics
//!
//! Errors are **delivered as events**, not returned: a node failure, a dead-end,
//! or an exhausted step budget produces a terminal [`StreamEvent::WorkflowError`],
//! keeping the stream's item type uniform. A dropped receiver simply stops the
//! driver early (all sends tolerate a closed channel).

use crate::core::state_graph::StateGraph;
use crate::{AgentNode, AgentState, LangGraphError};
use futures::future::join_all;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{Sender, channel};
use tokio_stream::wrappers::ReceiverStream;

/// Capacity of the bounded event channel.
///
/// Provides natural backpressure: if the consumer is slow, the background
/// driver task pauses on `send` instead of buffering unboundedly.
const STREAM_CHANNEL_CAPACITY: usize = 32;

/// Events emitted during graph execution streaming.
///
/// Each variant represents a distinct point in the workflow execution lifecycle,
/// allowing callers to observe progress, debug routing decisions, and monitor
/// per-node performance.
///
/// # Type Parameters
///
/// - `S`: The state type. Must implement [`AgentState`] + `Send` + `Sync`.
#[derive(Debug)]
pub enum StreamEvent<S: AgentState + Send + Sync> {
    /// The workflow has started execution.
    WorkflowStarted,

    /// A new step has begun. Nodes listed here run in parallel within this step.
    StepStarted {
        /// Step index (1-based), corresponding to one iteration of the execution loop.
        step: usize,
        /// Node names scheduled to run in parallel during this step.
        nodes: Vec<String>,
    },

    /// A single node has started execution.
    ///
    /// Emitted by the node's own concurrent future at the real moment it begins,
    /// so for parallel nodes these events interleave in true execution order.
    NodeStarted {
        /// Step index this node belongs to.
        step: usize,
        /// Node name.
        name: String,
    },

    /// A single node has finished execution.
    ///
    /// Emitted by the node's own concurrent future at the real moment it finishes;
    /// `elapsed` is this node's actual execution time, not the whole batch's.
    NodeFinished {
        /// Step index this node belongs to.
        step: usize,
        /// Node name.
        name: String,
        /// Time elapsed while executing this node.
        elapsed: Duration,
    },

    /// A routing decision has been resolved (static and/or conditional edges).
    RoutingDecision {
        /// Step index at which the decision was made.
        step: usize,
        /// Source nodes whose outgoing edges were evaluated.
        from_nodes: Vec<String>,
        /// Target nodes selected for the next step.
        to_nodes: Vec<String>,
    },

    /// The workflow finished successfully.
    WorkflowFinished {
        /// Final shared state after execution.
        state: Arc<S>,
        /// Total number of steps executed.
        total_steps: usize,
        /// Total time elapsed for the whole workflow.
        elapsed: Duration,
    },

    /// The workflow terminated due to an error.
    WorkflowError {
        /// Shared state at the point of failure.
        state: Arc<S>,
        /// Step index at which the error occurred.
        step: usize,
        /// The error that halted execution.
        error: LangGraphError,
    },
}

/// Emit a [`StreamEvent::WorkflowError`] and return `Err(())` to stop the driver.
///
/// A send failure only means the receiver was dropped, in which case the driver
/// intends to stop anyway; an error branch can therefore be written as a single
/// `return fail(..).await`.
async fn fail<S: AgentState + Send + Sync>(
    tx: &Sender<StreamEvent<S>>,
    state: &Arc<S>,
    step: usize,
    error: LangGraphError,
) -> Result<(), ()> {
    let _ = tx
        .send(StreamEvent::WorkflowError {
            state: Arc::clone(state),
            step,
            error,
        })
        .await;
    Err(())
}

/// Send a single event, returning `Err(())` if the receiver was dropped.
///
/// Used by [`run_driver`] for regular lifecycle events so that a dropped
/// receiver short-circuits execution via the `?` operator.
async fn emit<S: AgentState + Send + Sync>(
    tx: &Sender<StreamEvent<S>>,
    event: StreamEvent<S>,
) -> Result<(), ()> {
    tx.send(event).await.map_err(|_| ())
}

/// Run a single node, emitting [`StreamEvent::NodeStarted`] before and
/// [`StreamEvent::NodeFinished`] (with the node's real elapsed time) after.
async fn run_node_with_events<S: AgentState + Send + Sync>(
    tx: &Sender<StreamEvent<S>>,
    name: String,
    node: &dyn AgentNode<S>,
    state: Arc<S>,
    step: usize,
) -> Result<(), LangGraphError> {
    let _ = tx
        .send(StreamEvent::NodeStarted {
            step,
            name: name.clone(),
        })
        .await;
    let started = Instant::now();
    let result = node.apply(state).await;
    let _ = tx
        .send(StreamEvent::NodeFinished {
            step,
            name,
            elapsed: started.elapsed(),
        })
        .await;
    result
}

/// Execute multiple nodes concurrently, emitting node lifecycle events.
///
/// An event-aware variant of [`StateGraph::batch_apply`]: each node runs as a
/// concurrent future (via [`run_node_with_events`]) that emits its own
/// [`StreamEvent::NodeStarted`] / [`StreamEvent::NodeFinished`] into `tx` at the
/// real moment it starts and finishes, so events interleave in true execution
/// order and `elapsed` is the node's actual execution time (not the whole
/// batch's wall time).
///
/// # Arguments
///
/// * `tx` - Event channel sender.
/// * `node_names` - Names of the nodes, aligned with `nodes`.
/// * `nodes` - Node references to execute.
/// * `state` - Shared state passed to each node.
/// * `step` - Step index attached to the emitted events.
///
/// # Returns
///
/// - `Ok(())`: all nodes executed successfully.
/// - `Err(LangGraphError)`: first error from any node.
async fn batch_apply_with_events<S: AgentState + Send + Sync>(
    tx: &Sender<StreamEvent<S>>,
    node_names: &[String],
    nodes: Vec<&dyn AgentNode<S>>,
    state: Arc<S>,
    step: usize,
) -> Result<(), LangGraphError> {
    // Run all nodes concurrently; each emits its own start/finish events.
    // Node order is preserved so error reporting stays deterministic.
    let futures = node_names
        .iter()
        .zip(nodes)
        .map(|(name, node)| run_node_with_events(tx, name.clone(), node, Arc::clone(&state), step));
    for result in join_all(futures).await {
        result?; // Return first error immediately
    }
    Ok(())
}

/// Drive the workflow execution, pushing events into `tx`.
///
/// Returns `Ok(())` once [`StreamEvent::WorkflowFinished`] has been emitted, or
/// `Err(())` when execution stops early — either because a [`StreamEvent::WorkflowError`]
/// was reported or because the receiver was dropped. The return value carries no
/// information; it only drives the `?`-based control flow.
async fn run_driver<S: AgentState + Send + Sync + 'static>(
    graph: Arc<StateGraph<S>>,
    state: Arc<S>,
    tx: Sender<StreamEvent<S>>,
) -> Result<(), ()> {
    let start_time = Instant::now();

    let mut current = graph.start_nodes.clone();

    let mut step_count: usize = 0;
    let max_steps = graph.max_steps;
    let mut reached_end = false;

    emit(&tx, StreamEvent::WorkflowStarted).await?;

    loop {
        if step_count >= max_steps {
            break;
        }
        step_count += 1;

        if graph.is_end_node(&current) {
            current.remove(&graph.end_node);
            reached_end = true;
        }
        if current.is_empty() {
            break;
        }
        
        let node_names: Vec<String> = current.iter().cloned().collect();
        let nodes = match graph.get_node_by_keys(&current) {
            Ok(n) => n,
            Err(e) => return fail(&tx, &state, step_count, e).await,
        };
        if !nodes.is_empty() {
            emit(
                &tx,
                StreamEvent::StepStarted {
                    step: step_count,
                    nodes: node_names.clone(),
                },
            ).await?;
            if let Err(e) =
                batch_apply_with_events(&tx, &node_names, nodes, Arc::clone(&state), step_count)
                    .await
            {
                return fail(&tx, &state, step_count, e).await;
            }
        }

        let next = match graph
            .get_next_node_key(&current, state.as_ref())
        {
            Ok(n) => n,
            Err(e) => return fail(&tx, &state, step_count, e).await,
        };

        emit(
            &tx,
            StreamEvent::RoutingDecision {
                step: step_count,
                from_nodes: current.iter().cloned().collect(),
                to_nodes: next.iter().cloned().collect(),
            },
        )
            .await?;

        current = next;
    }

    if !reached_end {
        return fail(
            &tx,
            &state,
            step_count,
            LangGraphError::GraphError(format!(
                "Reached max_steps ({}) without hitting the end node; the graph likely contains a cycle",
                max_steps
            )),
        )
        .await;
    }

    let _ = tx
        .send(StreamEvent::WorkflowFinished {
            state: Arc::clone(&state),
            total_steps: step_count,
            elapsed: start_time.elapsed(),
        })
        .await;

    Ok(())
}

impl<S: AgentState + Send + Sync + 'static> StateGraph<S> {
    /// Execute the workflow graph and stream execution events in real time.
    ///
    /// Unlike [`invoke()`](StateGraph::invoke), which runs to completion before
    /// returning, this method spawns a background driver task and returns a
    /// [`Stream`](futures::Stream) that yields [`StreamEvent`]s as the workflow
    /// progresses.
    ///
    /// # Push-based design
    ///
    /// The architecture is **push-based** (mirroring Python LangGraph's queue
    /// model): nodes within a step run as **concurrent futures**, and each one
    /// emits its own [`StreamEvent::NodeStarted`] /
    /// [`StreamEvent::NodeFinished`] into a shared bounded channel at the real
    /// moment it starts and finishes. As a result:
    ///
    /// - `NodeStarted` / `NodeFinished` events interleave in **true execution
    ///   order**, even for parallel nodes;
    /// - `NodeFinished.elapsed` is each node's **actual** execution time, not the
    ///   whole batch's wall time;
    /// - the bounded channel provides natural backpressure.
    ///
    /// # Arguments
    ///
    /// * `state` - Initial shared state for this execution.
    ///
    /// # Returns
    ///
    /// A [`Stream`](futures::Stream) of [`StreamEvent`]s. The stream ends after
    /// emitting [`StreamEvent::WorkflowFinished`] on success, or a single
    /// [`StreamEvent::WorkflowError`] on failure.
    ///
    /// # Ownership
    ///
    /// This method takes `self: Arc<Self>` so the background task can own a
    /// reference to the graph with a `'static` lifetime. Wrap the compiled graph
    /// in an [`Arc`] before calling:
    ///
    /// ```rust
    /// use langgraph4rust::*;
    /// use std::collections::HashSet;
    /// use std::sync::Arc;
    ///
    /// #[derive(Clone)]
    /// struct HelloNode;
    ///
    /// #[async_trait]
    /// impl AgentNode<DefaultMemoryState> for HelloNode {
    ///     async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
    ///         state.set("output", "hello").await?;
    ///         Ok(())
    ///     }
    /// }
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), LangGraphError> {
    ///     let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    ///     builder.add_node("hello", Box::new(HelloNode));
    ///     builder.add_edge(START_NODE, HashSet::from(["hello".to_string()]));
    ///     builder.add_edge("hello", HashSet::from([END_NODE.to_string()]));
    ///     let graph = Arc::new(builder.compile()?);
    ///
    ///     let state = Arc::new(DefaultMemoryState::new());
    ///     let mut events = graph.stream(state);
    ///     while let Some(event) = events.next().await {
    ///         match event {
    ///             StreamEvent::WorkflowStarted => println!("workflow started"),
    ///             StreamEvent::StepStarted { step, nodes } => println!("step {}: {:?}", step, nodes),
    ///             StreamEvent::NodeStarted { step, name } => println!("step {}: node {} started", step, name),
    ///             StreamEvent::NodeFinished { step, name, elapsed } => {
    ///                 println!("step {}: node {} finished in {:?}", step, name, elapsed)
    ///             }
    ///             StreamEvent::RoutingDecision { step, from_nodes, to_nodes } => {
    ///                 println!("step {}: route {:?} -> {:?}", step, from_nodes, to_nodes)
    ///             }
    ///             StreamEvent::WorkflowFinished { total_steps, elapsed, .. } => {
    ///                 println!("finished {} steps in {:?}", total_steps, elapsed)
    ///             }
    ///             StreamEvent::WorkflowError { step, error, .. } => {
    ///                 println!("step {}: error {}", step, error)
    ///             }
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Error Handling
    ///
    /// Errors are delivered as a [`StreamEvent::WorkflowError`] event (the final
    /// item) rather than being returned, keeping the stream's item type uniform.
    pub fn stream(self: Arc<Self>, state: Arc<S>) -> ReceiverStream<StreamEvent<S>> {
        let (tx, rx) = channel::<StreamEvent<S>>(STREAM_CHANNEL_CAPACITY);

        tokio::spawn(async move {
            let _ = run_driver(self, state, tx).await;
        });

        ReceiverStream::new(rx)
    }
}
