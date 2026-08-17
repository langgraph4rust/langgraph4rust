//! Declarative, validated construction of workflow graphs.
//!
//! This module provides [`StateGraphBuilder`], the entry point for defining a
//! workflow. The builder exposes a fluent API for registering nodes and edges,
//! then [`compile`](StateGraphBuilder::compile)s the definition into an
//! immutable [`StateGraph`] after running comprehensive structural validation.
//!
//! # Virtual boundary nodes
//!
//! - [`START_NODE`] (`__start__`) — the implicit entry point; add edges *from*
//!   it to your real starting nodes.
//! - [`END_NODE`] (`__end__`) — the implicit terminator; add edges *to* it from
//!   your final nodes.
//!
//! # Edge types
//!
//! - **Static** ([`add_edge`](StateGraphBuilder::add_edge)) — deterministic,
//!   fixed targets.
//! - **Conditional** ([`add_conditional_edge`](StateGraphBuilder::add_conditional_edge))
//!   — synchronous router closures that pick the next node(s) from the runtime
//!   state. A node may use **either** static or conditional edges, not both.
//!
//! # Validation
//!
//! [`compile`](StateGraphBuilder::compile) consumes the builder and verifies the
//! topology (see its documentation for the full rule list), guaranteeing that a
//! successfully compiled graph is structurally sound before any execution.

use std::collections::{HashMap, HashSet};

use super::state_graph::StateGraph;
use crate::core::RouterFn;
use crate::core::agent_node::AgentNode;
use crate::core::agent_state::AgentState;
use crate::core::error::LangGraphError;
use crate::core::graph_validator::{GraphValidator, ValidatedGraph};

/// Special node name representing the entry point of the workflow.
///
/// All workflows begin execution from this virtual node. You don't need to
/// explicitly add it; just create edges from `START_NODE` to your actual
/// starting nodes.
///
/// # Example
///
/// ```rust
/// use langgraph4rust::*;
/// use std::collections::HashSet;
/// use std::sync::Arc;
///
/// #[derive(Clone)]
/// struct MyStartNode;
///
/// #[async_trait]
/// impl AgentNode<DefaultMemoryState> for MyStartNode {
///     async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
///         Ok(())
///     }
/// }
///
/// let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
/// builder.add_node("my_start_node", Box::new(MyStartNode));
///
/// // Connect START_NODE to your first real node
/// builder.add_edge(START_NODE, HashSet::from(["my_start_node".to_string()]));
/// ```
pub const START_NODE: &str = "__start__";

/// Special node name representing the exit point of the workflow.
///
/// When a node's edges lead to `END_NODE`, the workflow completes after that
/// node finishes executing. Multiple nodes can point to `END_NODE`.
///
/// # Example
///
/// ```rust
/// use langgraph4rust::*;
/// use std::collections::HashSet;
/// use std::sync::Arc;
///
/// #[derive(Clone)]
/// struct FinalStepNode;
///
/// #[async_trait]
/// impl AgentNode<DefaultMemoryState> for FinalStepNode {
///     async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
///         Ok(())
///     }
/// }
///
/// let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
/// builder.add_node("final_step", Box::new(FinalStepNode));
///
/// // Connect final step to END_NODE to complete the workflow
/// builder.add_edge("final_step", HashSet::from([END_NODE.to_string()]));
/// ```
pub const END_NODE: &str = "__end__";

/// Builder for constructing and validating workflow graphs.
///
/// This is the primary interface for defining workflow structures. Use this builder
/// to add nodes, define edges (both static and conditional), configure execution
/// parameters, and ultimately compile a ready-to-execute [`StateGraph`].
///
/// # Type Parameters
///
/// - `S`: The state type for all nodes in this graph. Must implement [`AgentState`].
///
/// # Lifecycle
///
/// 1. **Construction**: Create with [`StateGraphBuilder::new()`]
/// 2. **Configuration**: Add nodes, edges, set parameters
/// 3. **Compilation**: Call [`compile()`](StateGraphBuilder::compile) to validate and build
/// 4. **Execution**: Use the resulting [`StateGraph`] with [`StateGraph::invoke()`]
///
/// # Builder Pattern
///
/// Most methods return `&mut self` to allow method chaining:
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
/// builder.set_max_steps(100)
///     .add_node("process", Box::new(MyNode))
///     .add_edge(START_NODE, HashSet::from(["process".to_string()]))
///     .add_edge("process", HashSet::from([END_NODE.to_string()]));
/// let graph = builder.compile()?;
/// # Ok(())
/// # }
/// ```
///
/// # Validation
///
/// The [`compile()`](StateGraphBuilder::compile) method performs comprehensive validation:
/// - Ensures no empty graphs
/// - Verifies all edge targets exist as registered nodes
/// - Checks connectivity from start to end
/// - Detects isolated/disconnected components
/// - Warns about potential infinite loops (if max_steps not set)
///
/// # Example - Complete Workflow
///
/// ```rust
/// use langgraph4rust::*;
/// use std::collections::HashSet;
/// use std::sync::Arc;
///
/// #[derive(Clone)]
/// struct StartNode;
///
/// #[async_trait]
/// impl AgentNode<DefaultMemoryState> for StartNode {
///     async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
///         println!("Starting workflow");
///         Ok(())
///     }
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<(), LangGraphError> {
///     let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
///     builder.add_node("start", Box::new(StartNode));
///     builder.add_edge(START_NODE, HashSet::from(["start".to_string()]));
///     builder.add_edge("start", HashSet::from([END_NODE.to_string()]));
///
///     let graph = builder.compile()?;
///     let state = Arc::new(DefaultMemoryState::new());
///     graph.invoke(state).await?;
///     Ok(())
/// }
/// ```
pub struct StateGraphBuilder<S: AgentState + Send + Sync> {
    /// Collection of named nodes in the graph
    nodes: HashMap<String, Box<dyn AgentNode<S>>>,
    /// Static edges mapping source -> set of destinations
    edges: HashMap<String, HashSet<String>>,
    /// Conditional edges mapping source -> list of router functions
    conditional_edges: HashMap<String, Vec<RouterFn<S>>>,
    /// Start node set (defaults to {START_NODE})
    start_nodes: HashSet<String>,
    /// End node identifier (defaults to END_NODE)
    end_node: String,
    /// Maximum execution steps before forced termination
    max_steps: usize,
}

impl<S: AgentState + Send + Sync> Default for StateGraphBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: AgentState + Send + Sync> StateGraphBuilder<S> {
    /// Create a new empty graph builder with default configuration.
    ///
    /// Initializes a builder ready for workflow definition:
    /// - No nodes or edges defined yet
    /// - Default start node: [`START_NODE`] ("__start__")
    /// - Default end node: [`END_NODE`] ("__end__")
    /// - Maximum steps: `usize::MAX` (unlimited)
    ///
    /// # Returns
    ///
    /// A fresh `StateGraphBuilder` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use langgraph4rust::{StateGraphBuilder, DefaultMemoryState};
    ///
    /// let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// // Ready to configure...
    /// ```
    pub fn new() -> Self {
        StateGraphBuilder {
            max_steps: usize::MAX,
            nodes: Default::default(),
            edges: Default::default(),
            conditional_edges: Default::default(),
            start_nodes: HashSet::from([START_NODE.to_string()]),
            end_node: END_NODE.to_string(),
        }
    }

    /// Set the maximum number of execution steps before forced termination.
    ///
    /// This safety mechanism prevents infinite loops in cyclic graphs or
    /// graphs with complex routing logic. When the step count reaches this
    /// limit, execution stops with an error.
    ///
    /// # Arguments
    ///
    /// * `max_steps` - Maximum number of node executions allowed
    ///   - Use a reasonable number based on expected complexity (e.g., 100-1000)
    ///   - Set to `usize::MAX` effectively disables the limit (default behavior)
    ///
    /// # Returns
    ///
    /// `&mut self` for method chaining.
    ///
    /// # When to Use
    ///
    /// - **Always recommended** for graphs with cycles or conditional routing
    /// - **Optional** for simple linear/acyclic workflows
    /// - **Mandatory** if you suspect potential infinite loops
    ///
    /// # Example
    ///
    /// ```rust
    /// # use langgraph4rust::*;
    /// let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.set_max_steps(100);  // Limit to 100 steps
    /// ```
    pub fn set_max_steps(&mut self, max_steps: usize) -> &mut Self {
        self.max_steps = max_steps;
        self
    }

    /// Add a named node to the graph.
    ///
    /// Nodes are the executable units of your workflow. Each node must have
    /// a unique name and implement the [`AgentNode`] trait.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for this node (used in edge definitions)
    /// * `node` - Boxed [`AgentNode`] implementation containing the execution logic
    ///
    /// # Returns
    ///
    /// `&mut self` for method chaining.
    ///
    /// # Naming Rules
    ///
    /// - Must be unique within the graph (duplicates will overwrite previous nodes)
    /// - Cannot be empty string
    /// - Can contain any UTF-8 characters (including spaces, emojis, etc.)
    /// - Case-sensitive: `"MyNode"` ≠ `"mynode"`
    ///
    /// # Common Patterns
    ///
    /// ```rust
    /// # use langgraph4rust::*;
    /// # use std::sync::Arc;
    /// // Simple struct-based node
    /// #[derive(Clone)]
    /// struct ProcessDataNode;
    ///
    /// #[async_trait]
    /// impl AgentNode<DefaultMemoryState> for ProcessDataNode {
    ///     async fn apply(&self, state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> {
    ///         // Processing logic here...
    ///         Ok(())
    ///     }
    /// }
    ///
    /// // Add to builder
    /// # let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.add_node("process_data", Box::new(ProcessDataNode));
    /// ```
    ///
    /// # Note on Ownership
    ///
    /// The node is moved into the builder (via `Box`). After calling `compile()`,
    /// ownership transfers to the resulting [`StateGraph`].
    pub fn add_node(&mut self, name: &str, node: Box<dyn AgentNode<S>>) -> &mut Self {
        self.nodes.insert(name.to_string(), node);
        self
    }

    /// Add a static edge from one node to multiple target nodes.
    ///
    /// Static edges always connect to the same destination(s). Use this when
    /// the flow path is deterministic and doesn't depend on runtime state.
    ///
    /// # Arguments
    ///
    /// * `from` - Source node name (must exist via [`add_node()`](StateGraphBuilder::add_node))
    /// * `to` - Set of target node names (all must exist when compiled)
    ///
    /// # Returns
    ///
    /// `&mut self` for method chaining.
    ///
    /// # Edge Types
    ///
    /// **Single Target**: One-to-one connection
    /// ```rust
    /// # use langgraph4rust::*;
    /// # use std::collections::HashSet;
    /// # let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.add_edge("step1", HashSet::from(["step2".to_string()]));
    /// ```
    ///
    /// **Multiple Targets**: Fan-out pattern (targets execute in parallel)
    /// ```rust
    /// # use langgraph4rust::*;
    /// # use std::collections::HashSet;
    /// # let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.add_edge("router", HashSet::from([
    ///     "process_a".to_string(),
    ///     "process_b".to_string(),
    ///     "process_c".to_string(),
    /// ]));
    /// ```
    ///
    /// **Special Nodes**: Connect to START/END
    /// ```rust
    /// # use langgraph4rust::*;
    /// # use std::collections::HashSet;
    /// # let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.add_edge(START_NODE, HashSet::from(["entry".to_string()]));
    /// builder.add_edge("final", HashSet::from([END_NODE.to_string()]));
    /// ```
    ///
    /// # Validation
    ///
    /// Edge targets are validated during [`compile()`](StateGraphBuilder::compile):
    /// - All target node names must exist in the graph
    /// - Source node must exist (or be special START/END node)
    pub fn add_edge(&mut self, from: &str, to: HashSet<String>) -> &mut Self {
        self.edges.insert(from.to_string(), to);
        self
    }

    /// Add a conditional edge with dynamic routing based on state.
    ///
    /// Unlike static edges, conditional edges use **router functions** to determine
    /// the next node(s) at runtime based on current state values. This enables
    /// powerful dynamic workflows like:
    /// - If-else branching
    /// - Loop until condition met
    /// - Dynamic dispatch patterns
    ///
    /// # Arguments
    ///
    /// * `from` - Source node name
    /// * `routers` - Vector of routing functions, each returning a target node name
    ///
    /// # Router Function Signature
    ///
    /// Each router has signature: `Fn(&S) -> String`
    /// - Receives read-only reference to current state
    /// - Returns the **name of the next node** to execute
    /// - Called at runtime during graph execution
    ///
    /// # Returns
    ///
    /// `&mut self` for method chaining.
    ///
    /// # Example - Conditional Branching
    ///
    /// ```rust,ignore
    /// # use langgraph4rust::*;
    /// # use std::sync::Arc;
    /// # let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.add_conditional_edge("decision_point", vec![
    ///     // Route based on state value
    ///     Box::new(|state| {
    ///         let should_process: bool = state.get("flag").ok().flatten().unwrap_or(false);
    ///         if should_process { "process".to_string() } else { "skip".to_string() }
    ///     }),
    /// ]);
    /// ```
    ///
    /// # Example - Loop Pattern
    ///
    /// ```rust,ignore
    /// # use langgraph4rust::*;
    /// # use std::sync::Arc;
    /// # let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.add_conditional_edge("retry_check", vec![
    ///     Box::new(|state| {
    ///         let attempts: i32 = state.get("attempts").ok().flatten().unwrap_or(0);
    ///         if attempts < 3 { "retry".to_string() } else { "give_up".to_string() }
    ///     }),
    /// ]);
    /// ```
    ///
    /// # Important Notes
    ///
    /// - **Router must return valid node names** (exist in the graph)
    /// - **Multiple routers**: Only first non-error result is used (future: support multiple)
    /// - **State access**: Routers receive `&S` (read-only), cannot modify state
    /// - **Error handling**: Panics in routers become `LangGraphError`
    ///
    /// # Validation
    ///
    /// During compilation, the system verifies routers return valid node names
    /// where possible (static analysis). Runtime validation catches invalid returns.
    pub fn add_conditional_edge(&mut self, from: &str, routers: Vec<RouterFn<S>>) -> &mut Self {
        self.conditional_edges.insert(from.to_string(), routers);
        self
    }

    /// Override the default start node for this graph.
    ///
    /// By default, all workflows begin at [`START_NODE`] (`"__start__"`).
    /// Use this method to specify a different entry point if needed.
    /// This replaces any previously set start nodes.
    ///
    /// # Arguments
    ///
    /// * `start_node` - Name of the node that should serve as the entry point
    ///
    /// # Returns
    ///
    /// `&mut self` for method chaining.
    ///
    /// # When to Use
    ///
    /// - Customizing workflow entry points
    /// - Testing specific subgraphs
    /// - Creating reusable graph fragments
    ///
    /// # Note
    ///
    /// The specified node must still be added via [`add_node()`](StateGraphBuilder::add_node).
    /// This only changes which node receives initial control flow.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use langgraph4rust::*;
    /// # use std::sync::Arc;
    /// # #[derive(Clone)]
    /// # struct EntryNode;
    /// # #[async_trait]
    /// # impl AgentNode<DefaultMemoryState> for EntryNode {
    /// #     async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> { Ok(()) }
    /// # }
    /// let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.add_node("custom_entry", Box::new(EntryNode));
    /// builder.set_start_node("custom_entry");
    /// ```
    pub fn set_start_node(&mut self, start_node: &str) -> &mut Self {
        self.start_nodes = HashSet::from([start_node.to_string()]);
        self
    }

    /// Add an additional start node to the graph.
    ///
    /// Multiple start nodes allow batch execution or replay from multiple
    /// entry points. The default start node ([`START_NODE`]) is always
    /// included unless replaced via [`set_start_node()`](Self::set_start_node).
    ///
    /// # Arguments
    ///
    /// * `start_node` - Name of an additional entry point node
    ///
    /// # Returns
    ///
    /// `&mut self` for method chaining.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use langgraph4rust::*;
    /// # use std::sync::Arc;
    /// # #[derive(Clone)]
    /// # struct EntryA;
    /// # #[derive(Clone)]
    /// # struct EntryB;
    /// # #[async_trait]
    /// # impl AgentNode<DefaultMemoryState> for EntryA {
    /// #     async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> { Ok(()) }
    /// # }
    /// # #[async_trait]
    /// # impl AgentNode<DefaultMemoryState> for EntryB {
    /// #     async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> { Ok(()) }
    /// # }
    /// let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.add_node("entry_a", Box::new(EntryA))
    ///     .add_node("entry_b", Box::new(EntryB))
    ///     .set_start_node("entry_a")
    ///     .add_start_node("entry_b");
    /// ```
    pub fn add_start_node(&mut self, start_node: &str) -> &mut Self {
        self.start_nodes.insert(start_node.to_string());
        self
    }

    /// Override the default end node for this graph.
    ///
    /// By default, workflows end when reaching [`END_NODE`] (`"__end__"`).
    /// Use this to customize termination conditions.
    ///
    /// # Arguments
    ///
    /// * `end_node` - Name of the node that signals workflow completion
    ///
    /// # Returns
    ///
    /// `&mut self` for method chaining.
    ///
    /// # When to Use
    ///
    /// - Custom exit points for subgraphs
    /// - Named termination states
    /// - Integration with external systems
    ///
    /// # Example
    ///
    /// ```rust
    /// # use langgraph4rust::*;
    /// let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    /// builder.set_end_node("completion_handler");
    /// ```
    pub fn set_end_node(&mut self, end_node: &str) -> &mut Self {
        self.end_node = end_node.to_string();
        self
    }

    /// Compile the graph definition into an executable [`StateGraph`].
    ///
    /// This is the final step in building a workflow. It consumes the builder,
    /// performs comprehensive validation, and produces an immutable, ready-to-execute
    /// graph instance.
    ///
    /// # Validation Steps
    ///
    /// The compilation process checks for:
    ///
    /// 1. **Empty Graph Error**
    ///    - At least one node must be defined
    ///
    /// 2. **Start Node Connectivity**
    ///    - Start node must have outgoing edges (unless it's the only node)
    ///
    /// 3. **Edge Target Existence**
    ///    - All edge targets must refer to registered nodes
    ///    - Both static and conditional edges are checked
    ///
    /// 4. **End Node Reachability**
    ///    - There must be a path from start to end node
    ///    - Prevents dead-end workflows
    ///
    /// 5. **Isolated Node Detection**
    ///    - Warns about nodes unreachable from start
    ///    - Warns about nodes that can't reach end
    ///
    /// 6. **Cycle Safety Check**
    ///    - If graph contains cycles, requires `max_steps` to be set
    ///    - Prevents accidental infinite loops
    ///
    /// 7. **Edge Type Mutual Exclusivity**
    ///    - A node cannot have both static edges and conditional edges
    ///    - To merge multiple routing targets, use a single conditional edge
    ///      with multiple router functions (their results are unioned)
    ///
    /// # Returns
    ///
    /// - **Ok(StateGraph)**: Successfully compiled, ready for execution
    /// - **Err(LangGraphError)**: Validation failed with descriptive error message
    ///
    /// # Errors
    ///
    /// Returns [`LangGraphError::GraphError`] for structural issues:
    ///
    /// ```text
    /// Graph error: Empty graph - at least one node required
    /// Graph error: Edge from 'node_a' references unknown target 'node_z'
    /// Graph error: No path exists from '__start__' to '__end__'
    /// Graph error: Node 'node_a' cannot have both static edges and conditional edges
    /// ```
    ///
    /// # Consumption
    ///
    /// **Important**: This method consumes `self`. After calling `compile()`,
    /// you cannot modify the builder further. This ensures the compiled graph
    /// is immutable and safe for concurrent execution.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use langgraph4rust::*;
    /// # use std::collections::HashSet;
    /// # use std::sync::Arc;
    /// # #[derive(Clone)]
    /// # struct MyNode;
    /// # #[async_trait]
    /// # impl AgentNode<DefaultMemoryState> for MyNode {
    /// #     async fn apply(&self, _state: Arc<DefaultMemoryState>) -> Result<(), LangGraphError> { Ok(()) }
    /// # }
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), LangGraphError> {
    /// let mut builder = StateGraphBuilder::<DefaultMemoryState>::new();
    ///
    /// // Configure the graph...
    /// builder.add_node("my_node", Box::new(MyNode));
    /// builder.add_edge(START_NODE, HashSet::from(["my_node".to_string()]));
    /// builder.add_edge("my_node", HashSet::from([END_NODE.to_string()]));
    ///
    /// // Compile and get executable graph
    /// match builder.compile() {
    ///     Ok(graph) => {
    ///         let state = Arc::new(DefaultMemoryState::new());
    ///         graph.invoke(state).await?;
    ///     }
    ///     Err(e) => {
    ///         eprintln!("Failed to compile: {}", e);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance
    ///
    /// Compilation is O(N + E) where N = nodes, E = edges. For typical workflows
    /// (< 100 nodes), this completes in microseconds.
    pub fn compile(self) -> Result<StateGraph<S>, LangGraphError> {
        let validator = GraphValidator {
            max_steps: self.max_steps,
            nodes: self.nodes,
            edges: self.edges,
            conditional_edges: self.conditional_edges,
            start_nodes: self.start_nodes,
            end_node: self.end_node,
        };

        let validated: ValidatedGraph<S> = validator.validate()?;

        Ok(StateGraph::new(
            validated.max_steps,
            validated.nodes,
            validated.edges,
            validated.conditional_edges,
            validated.start_nodes,
            validated.end_node,
        ))
    }
}