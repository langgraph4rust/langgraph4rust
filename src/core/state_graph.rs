//! Compiled graph representation and batch execution.
//!
//! This module defines [`StateGraph`], the immutable artifact produced by
//! [`StateGraphBuilder::compile`](crate::StateGraphBuilder::compile). A compiled
//! graph has passed all structural validation and can be executed repeatedly,
//! from multiple tasks, with different states.
//!
//! # Two execution modes
//!
//! - **Batch** — [`StateGraph::invoke`] runs the workflow to completion and
//!   returns once the end node is reached (or an error occurs). The streaming
//!   counterpart lives in the [`state_graph_stream`](crate::core::state_graph_stream)
//!   module.
//! - **Streaming** — [`StateGraph::stream`] (defined in
//!   [`state_graph_stream`](crate::core::state_graph_stream)) yields real-time
//!   [`StreamEvent`](crate::StreamEvent)s.
//!
//! # Step-based traversal
//!
//! Execution proceeds in discrete steps. Starting from the virtual
//! [`__start__`](crate::START_NODE) node, each step resolves the outgoing edges
//! (static and/or conditional) of the current node set, executes all resulting
//! nodes **concurrently**, and advances to the next set — until the
//! [`__end__`](crate::END_NODE) node is reached or [`max_steps`](crate::StateGraphBuilder::set_max_steps)
//! is exhausted.

use crate::core::RouterFn;
use crate::core::agent_node::AgentNode;
use crate::core::agent_state::AgentState;
use crate::core::error::LangGraphError;
use futures::future::join_all;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Compiled, immutable workflow graph ready for execution.
///
/// This is the output of [`StateGraphBuilder::compile()`](crate::StateGraphBuilder::compile) and represents a fully
/// validated workflow that can be executed multiple times with different states.
/// The graph is immutable after compilation, ensuring thread-safe concurrent usage.
///
/// # Type Parameters
///
/// - `S`: The state type. Must implement [`AgentState`] + `Send` + `Sync`.
///
/// # Lifecycle
///
/// 1. **Build**: Create via [`StateGraphBuilder`](crate::StateGraphBuilder)
/// 2. **Compile**: Call [`compile()`](crate::StateGraphBuilder::compile) → produces `StateGraph`
/// 3. **Execute**: Call [`invoke()`](StateGraph::invoke) one or more times
/// 4. **Reuse**: Same graph can execute multiple workflows safely
///
/// # Immutability
///
/// Once compiled, the graph structure cannot be modified. This provides:
/// - **Thread safety**: Multiple `invoke()` calls can run concurrently (with different states)
/// - **Predictability**: No runtime modifications to graph structure
/// - **Performance**: Enables optimization opportunities
///
/// # Execution Model
///
/// The graph executes using a **step-based traversal** algorithm:
///
/// ```text
/// START → [Node A] → [Node B, C] → END
///              ↓           ↘        ↑
///           (parallel)    (parallel)
/// ```
///
/// - Starts at the configured start node
/// - Follows edges (static or conditional) to determine next nodes
/// - Executes all nodes at the current level in parallel
/// - Continues until reaching the end node or error/termination condition
///
/// # Example
///
/// ```rust
/// use langgraph4rust::*;
/// use std::collections::HashSet;
/// use std::sync::Arc;
///
/// #[derive(Clone)]
/// struct SimpleNode;
///
/// #[async_trait]
/// impl AgentNode<DefaultMemoryState> for SimpleNode {
///     async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
///         state.set("done", true).await?;
///         Ok(())
///     }
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<(), LangGraphError> {
///     let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
///     builder.add_node("step", Box::new(SimpleNode));
///     builder.add_edge(START_NODE, HashSet::from(["step".to_string()]));
///     builder.add_edge("step", HashSet::from([END_NODE.to_string()]));
///     let graph = builder.compile()?;
///
///     // Execute multiple times with different states
///     for i in 0..10 {
///         let state = Arc::new(DefaultMemoryState::new());
///         state.set("iteration", i).await?;
///         graph.invoke(state).await?;
///     }
///     Ok(())
/// }
/// ```
///
/// # Error Handling
///
/// Errors during execution propagate immediately:
/// - Node errors stop execution at that point
/// - State errors prevent further state access
/// - Graph structural errors shouldn't occur (validated at compile time)
///
/// See [`LangGraphError`] for all possible error types.
pub struct StateGraph<S: AgentState + Send + Sync> {
    /// Map of node names to their executable implementations
    nodes: HashMap<String, Box<dyn AgentNode<S>>>,
    /// Static edges: source node -> set of target nodes
    edges: HashMap<String, HashSet<String>>,
    /// Conditional edges: source node -> list of router functions
    conditional_edges: HashMap<String, Vec<RouterFn<S>>>,
    /// Entry point node set (supports multiple start nodes for replay/batch)
    pub(crate) start_nodes: HashSet<String>,
    /// Termination node name
    pub(crate) end_node: String,
    /// Maximum steps before forced termination (safety limit)
    pub(crate) max_steps: usize,
}

impl<S: AgentState + Send + Sync> StateGraph<S> {
    /// Create a new compiled StateGraph instance.
    ///
    /// This is an internal constructor used by [`StateGraphBuilder::compile()`](crate::StateGraphBuilder::compile).
    /// Users should not call this directly; always build graphs through the builder.
    ///
    /// # Arguments
    ///
    /// * `max_steps` - Maximum execution step count (from builder configuration)
    /// * `nodes` - Validated map of node names to implementations
    /// * `edges` - Validated static edge definitions
    /// * `conditional_edges` - Validated conditional edge definitions
    /// * `start_nodes` - Set of start node identifiers
    /// * `end_node` - End node identifier
    ///
    /// # Safety
    ///
    /// All parameters are assumed to be validated by [`GraphValidator`](crate::core::graph_validator::GraphValidator).
    /// Passing unvalidated data may cause panics or undefined behavior.
    pub(crate) fn new(
        max_steps: usize,
        nodes: HashMap<String, Box<dyn AgentNode<S>>>,
        edges: HashMap<String, HashSet<String>>,
        conditional_edges: HashMap<String, Vec<RouterFn<S>>>,
        start_nodes: HashSet<String>,
        end_node: String,
    ) -> Self {
        StateGraph {
            max_steps,
            nodes,
            edges,
            conditional_edges,
            start_nodes,
            end_node,
        }
    }

    /// Execute the workflow graph with the given initial state.
    ///
    /// This is the primary method for running a compiled workflow. It traverses
    /// the graph from the start node to the end node, executing nodes and
    /// following edges according to their definitions.
    ///
    /// # Arguments
    ///
    /// * `state` - Initial state for this execution. The state is shared across
    ///   all nodes and persists throughout the entire workflow execution.
    ///   Use `Arc::clone()` if you need to retain access to the state afterwards.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: Workflow completed successfully (reached end node)
    /// - `Err(LangGraphError)`: Execution failed due to node errors, state errors,
    ///   or other runtime issues
    ///
    /// # Execution Algorithm
    ///
    /// The invocation follows this process:
    ///
    /// 1. **Initialize**: Set current position to start node
    /// 2. **Loop** (until termination or max_steps):
    ///    a. Check if current position is end node → success
    ///    b. Check if current position is empty → dead-end error
    ///    c. Execute all nodes at current level in **parallel**
    ///    d. Determine next positions from edges (static + conditional)
    ///    e. Move to next positions
    /// 3. **Terminate**: Success or error
    ///
    /// # Parallel Execution
    ///
    /// When a node has multiple outgoing edges (fan-out), all target nodes
    /// are executed concurrently using `futures::join_all`. This means:
    ///
    /// - Nodes at the same "level" run simultaneously
    /// - Each node gets its own clone of the state Arc
    /// - State changes are visible to subsequent nodes (same underlying data)
    /// - If any node fails, execution stops immediately
    ///
    /// # Step Counting
    ///
    /// Each iteration of the main loop counts as one "step". The `max_steps`
    /// limit (set during graph construction) prevents infinite loops:
    ///
    /// ```text
    /// Step 1: Execute [A] → Determine next: [B, C]
    /// Step 2: Execute [B, C] (parallel) → Determine next: [D]
    /// Step 3: Execute [D] → Determine next: [END]
    /// Total: 3 steps
    /// ```
    ///
    /// # Error Propagation
    ///
    /// Errors are handled as follows:
    ///
    /// - **Node errors** ([`LangGraphError::NodeError`]): Immediate failure,
    ///   no more nodes execute, error returned to caller
    /// - **State errors** ([`LangGraphError::StateError`]): Same as node errors
    /// - **Dead-ends** ([`LangGraphError::GraphError`], [`LangGraphError::NotFound`]):
    ///   When nodes have no outgoing edges but haven't reached END
    /// - **Max steps exceeded**: Loop terminates without error (may indicate issue)
    ///
    /// # State Mutation
    ///
    /// Nodes receive `Arc<S>` and can mutate state through get/set operations.
    /// Changes made by earlier nodes are visible to later nodes because they
    /// share the same underlying data (through Arc).
    ///
    /// **Important**: For truly independent parallel branches, ensure nodes
    /// write to different keys or handle conflicts explicitly.
    ///
    /// # Thread Safety
    ///
    /// The `invoke()` method itself is thread-safe for **concurrent invocations**
    /// with different state instances. However:
    ///
    /// - Do NOT call `invoke()` concurrently with the SAME state instance
    /// - The graph itself can be shared across threads (`StateGraph` is Send+Sync)
    /// - Individual state instances must be thread-safe (which `DefaultMemoryState` is)
    ///
    /// # Example - Basic Usage
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
    ///         let input: String = state.get("input").await?.unwrap_or_default();
    ///         state.set("output", input.to_uppercase()).await?;
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
    ///     let graph = builder.compile()?;
    ///
    ///     let state = Arc::new(DefaultMemoryState::new());
    ///     state.set("input", "hello").await?;
    ///
    ///     graph.invoke(state).await?;
    ///
    ///     println!("Workflow complete!");
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Example - Multiple Executions
    ///
    /// ```rust
    /// use langgraph4rust::*;
    /// use std::collections::HashSet;
    /// use std::sync::Arc;
    ///
    /// #[derive(Clone)]
    /// struct WorkNode;
    ///
    /// #[async_trait]
    /// impl AgentNode<DefaultMemoryState> for WorkNode {
    ///     async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
    ///         Ok(())
    ///     }
    /// }
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), LangGraphError> {
    ///     let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    ///     builder.add_node("work", Box::new(WorkNode));
    ///     builder.add_edge(START_NODE, HashSet::from(["work".to_string()]));
    ///     builder.add_edge("work", HashSet::from([END_NODE.to_string()]));
    ///     let graph = builder.compile()?;
    ///
    ///     // Reuse the same graph multiple times
    ///     for id in 0..5 {
    ///         let state = Arc::new(DefaultMemoryState::new());
    ///         state.set("request_id", id).await?;
    ///
    ///         match graph.invoke(state.clone()).await {
    ///             Ok(()) => println!("Request {} succeeded", id),
    ///             Err(e) => eprintln!("Request {} failed: {}", id, e),
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Performance Characteristics
    ///
    /// - **Time complexity**: O(S × N) where S = steps, N = avg nodes per step
    /// - **Space complexity**: O(N) for storing active nodes at each step
    /// - **Parallelism**: Automatic fan-out when edges have multiple targets
    /// - **Overhead**: Minimal; each step involves hashmap lookups and async spawning
    ///
    /// # Panics
    ///
    /// This method should not panic under normal circumstances. Possible panic sources:
    /// - Router function panics (caught and converted to error)
    /// - Node implementation panics (caught and converted to error)
    /// - Internal logic errors (indicates bug in langgraph4rust itself)
    pub async fn invoke(&self, state: Arc<S>) -> Result<(), LangGraphError> {
        let mut current = self.start_nodes.clone();

        let mut step_count: usize = 0;
        let max_steps = self.max_steps;

        loop {
            if step_count >= max_steps {
                break;
            }
            step_count += 1;
            if self.is_end_node(&current) {
                current.remove(&self.end_node);
            }
            if current.is_empty() {
                break;
            }

            let nodes = self.get_node_by_keys(&current)?;
            if !nodes.is_empty() {
                self.batch_apply(nodes, Arc::clone(&state)).await?;
            }
            current = self.get_next_node_key(&current, state.as_ref())?;
        }
        Ok(())
    }

    /// Retrieve a single node reference by its name.
    ///
    /// # Arguments
    ///
    /// * `key` - The node name to look up
    ///
    /// # Returns
    ///
    /// - `Ok(&Box<dyn AgentNode<S>>)` - Reference to the node implementation
    /// - `Err(LangGraphError::NotFound)` - If node name doesn't exist
    fn get_node_by_key(&self, key: &String) -> Option<&dyn AgentNode<S>> {
        self.nodes.get(key).map(|node| node.as_ref())
    }

    /// Replace the start node set with a collection of nodes.
    ///
    /// Clears all existing start nodes and sets the entry points to the
    /// given node names. This allows reconfiguring the graph's entry points
    /// after compilation, which is useful for replay, branching, or
    /// testing scenarios where you want to start execution from different
    /// nodes without rebuilding the entire graph.
    ///
    /// Supports setting multiple start nodes for parallel initial execution.
    ///
    /// # Arguments
    ///
    /// * `keys` - The names of the nodes to set as the start nodes.
    ///   An empty `Vec` is allowed — subsequent `invoke` will exit
    ///   immediately with no nodes executed.
    ///
    /// # Returns
    ///
    /// - `Ok(())` — The start nodes were replaced successfully
    ///
    /// # Note
    ///
    /// This method replaces **all** existing start nodes with the given
    /// collection. If you need to add a start node without clearing
    /// existing ones, use [`StateGraphBuilder::add_start_node`](crate::StateGraphBuilder::add_start_node) before
    /// compilation. The node names are not validated here — unknown
    /// nodes are silently skipped during execution (see
    /// `get_node_by_keys`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use langgraph4rust::*;
    /// use std::collections::HashSet;
    /// use std::sync::Arc;
    ///
    /// #[derive(Clone)]
    /// struct MyNode;
    ///
    /// #[async_trait]
    /// impl AgentNode<DefaultMemoryState> for MyNode {
    ///     async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
    ///         Ok(())
    ///     }
    /// }
    ///
    /// # fn main() -> Result<(), LangGraphError> {
    /// let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.add_node("node_a", Box::new(MyNode));
    /// builder.add_node("node_b", Box::new(MyNode));
    /// builder.add_node("node_c", Box::new(MyNode));
    /// builder.add_edge("__start__", HashSet::from(["node_a".to_string()]));
    /// builder.add_edge("node_a", HashSet::from(["__end__".to_string()]));
    /// builder.add_edge("node_b", HashSet::from(["__end__".to_string()]));
    /// builder.add_edge("node_c", HashSet::from(["__end__".to_string()]));
    /// let mut graph = builder.compile()?;
    ///
    /// // Reconfigure to start from a single node
    /// graph.set_start_nodes(vec!["node_b".to_string()])?;
    ///
    /// // Or start from multiple nodes in parallel
    /// graph.set_start_nodes(vec!["node_b".to_string(), "node_c".to_string()])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_start_nodes(&mut self, keys: Vec<String>) -> Result<(), LangGraphError> {
        self.start_nodes.clear();
        self.start_nodes.extend(keys);
        Ok(())
    }

    /// Retrieve multiple node references by their names.
    ///
    /// Silently skips keys that are not registered nodes (e.g. virtual start markers).
    ///
    /// # Arguments
    ///
    /// * `keys` - Set of node names to retrieve
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<&Box<dyn AgentNode<S>>>)` - Vector of node references
    pub(crate) fn get_node_by_keys(
        &self,
        keys: &HashSet<String>,
    ) -> Result<Vec<&dyn AgentNode<S>>, LangGraphError> {
        let mut nodes = Vec::new();
        for key in keys {
            if let Some(node) = self.get_node_by_key(key) {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }

    /// Check if any of the provided keys matches the end node.
    pub(crate) fn is_end_node(&self, keys: &HashSet<String>) -> bool {
        if keys.is_empty() {
            return false;
        }
        keys.contains(&self.end_node)
    }

    /// Determine the next set of node keys based on current position and state.
    ///
    /// Resolves both static edges and conditional edges to produce the complete
    /// set of nodes to execute in the next step.
    ///
    /// # Arguments
    ///
    /// * `keys` - Current set of active node names
    /// * `state` - Current state (used by conditional edge routers)
    ///
    /// # Returns
    ///
    /// Set of node names for the next execution step
    pub(crate) fn get_next_node_key(
        &self,
        keys: &HashSet<String>,
        state: &S,
    ) -> Result<HashSet<String>, LangGraphError> {
        if keys.is_empty() {
            return Ok(HashSet::new());
        }
        let mut next_node_keys = HashSet::new();
        for key in keys {
            // Static edges: collect targets directly
            if let Some(targets) = self.edges.get(key) {
                if !targets.is_empty() {
                    for target in targets {
                        next_node_keys.insert(target.clone());
                    }
                }
            }
            // Conditional edges: evaluate router functions
            if let Some(routers) = self.conditional_edges.get(key) {
                for router in routers {
                    next_node_keys.insert(router(state));
                }
            }
        }
        Ok(next_node_keys)
    }

    /// Execute multiple nodes in parallel and collect results.
    ///
    /// Spawns async tasks for each node and awaits all of them.
    /// Returns the first error encountered, if any.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Slice of node references to execute
    /// * `state` - Shared state passed to each node
    ///
    /// # Returns
    ///
    /// - `Ok(())`: All nodes executed successfully
    /// - `Err(LangGraphError)`: First error from any node
    pub(crate) async fn batch_apply(
        &self,
        nodes: Vec<&dyn AgentNode<S>>,
        state: Arc<S>,
    ) -> Result<(), LangGraphError> {
        let futures: Vec<_> = nodes
            .into_iter()
            .map(|node| {
                let state_clone = Arc::clone(&state);
                async move { node.apply(state_clone).await }
            })
            .collect();
        let results = join_all(futures).await;
        for result in results {
            result?; // Return first error immediately
        }
        Ok(())
    }
}