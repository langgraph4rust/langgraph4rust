//! Compile-time structural validation of workflow graphs.
//!
//! This module hosts the validation pass that runs inside
//! [`StateGraphBuilder::compile`](crate::StateGraphBuilder::compile). It turns a
//! raw builder definition into a [`ValidatedGraph`] only after every structural
//! rule passes, guaranteeing that a compiled [`StateGraph`](crate::StateGraph)
//! is sound before execution.
//!
//! # Checks performed
//!
//! - `max_steps` is non-zero
//! - start / end node names are valid and distinct
//! - at least one node exists, with non-empty names
//! - the start node is not also a registered node and has outgoing edges
//! - every static edge target refers to a registered node (or the end node)
//! - every conditional edge source is a registered node (or the start node)
//! - **no node has both static and conditional edges** (mutual exclusivity)
//!
//! Note that conditional edge *targets* (the values returned by routers) are
//! resolved at runtime, not here.

use std::collections::{HashMap, HashSet};

use crate::core::RouterFn;
use crate::core::agent_node::AgentNode;
use crate::core::agent_state::AgentState;
use crate::core::error::LangGraphError;

/// A graph definition that has passed all validation checks.
///
/// Produced by [`GraphValidator::validate`] and consumed by
/// [`StateGraphBuilder::compile`](crate::StateGraphBuilder::compile) to construct
/// the final [`StateGraph`](crate::StateGraph). Its fields mirror the builder's,
/// but their invariants are now guaranteed.
pub struct ValidatedGraph<S: AgentState + Send + Sync> {
    pub max_steps: usize,
    pub nodes: HashMap<String, Box<dyn AgentNode<S>>>,
    pub edges: HashMap<String, HashSet<String>>,
    pub conditional_edges: HashMap<String, Vec<RouterFn<S>>>,
    pub start_nodes: HashSet<String>,
    pub end_node: String,
}

/// Validates a raw graph definition before it becomes executable.
///
/// Constructed internally by
/// [`StateGraphBuilder::compile`](crate::StateGraphBuilder::compile); call
/// [`validate`](GraphValidator::validate) to run every structural check.
pub struct GraphValidator<S: AgentState + Send + Sync> {
    pub max_steps: usize,
    pub nodes: HashMap<String, Box<dyn AgentNode<S>>>,
    pub edges: HashMap<String, HashSet<String>>,
    pub conditional_edges: HashMap<String, Vec<RouterFn<S>>>,
    pub start_nodes: HashSet<String>,
    pub end_node: String,
}

impl<S: AgentState + Send + Sync> GraphValidator<S> {
    /// Run all structural validation checks.
    ///
    /// # Returns
    ///
    /// - `Ok(ValidatedGraph)` — every check passed; the definition is sound.
    /// - `Err(LangGraphError::GraphError)` — the first violated rule, with a
    ///   descriptive message.
    pub fn validate(self) -> Result<ValidatedGraph<S>, LangGraphError> {
        Self::validate_max_steps(self.max_steps)?;
        Self::validate_start_end_nodes(&self.start_nodes, &self.end_node)?;
        Self::validate_nodes_exist(&self.nodes)?;
        Self::validate_node_names_not_empty(&self.nodes)?;
        Self::validate_start_end_different(&self.start_nodes, &self.end_node)?;
        Self::validate_start_has_outgoing_edges(
            &self.start_nodes,
            &self.edges,
            &self.conditional_edges,
        )?;
        Self::validate_static_edges_valid(
            &self.edges,
            &self.nodes,
            &self.start_nodes,
            &self.end_node,
        )?;
        Self::validate_conditional_edges_valid(
            &self.conditional_edges,
            &self.nodes,
            &self.start_nodes,
        )?;
        Self::validate_no_mixed_edge_types(&self.edges, &self.conditional_edges)?;

        Ok(ValidatedGraph {
            max_steps: self.max_steps,
            nodes: self.nodes,
            edges: self.edges,
            conditional_edges: self.conditional_edges,
            start_nodes: self.start_nodes,
            end_node: self.end_node,
        })
    }

    fn validate_max_steps(max_steps: usize) -> Result<(), LangGraphError> {
        if max_steps == 0 {
            return Err(LangGraphError::GraphError(
                "max_steps must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_start_end_nodes(
        start_nodes: &HashSet<String>,
        end_node: &str,
    ) -> Result<(), LangGraphError> {
        if start_nodes.is_empty() {
            return Err(LangGraphError::GraphError(
                "Start node set cannot be empty".to_string(),
            ));
        }
        for start in start_nodes {
            if start.is_empty() {
                return Err(LangGraphError::GraphError(
                    "Start node cannot be empty".to_string(),
                ));
            }
        }
        if end_node.is_empty() {
            return Err(LangGraphError::GraphError(
                "End node cannot be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_nodes_exist(
        nodes: &HashMap<String, Box<dyn AgentNode<S>>>,
    ) -> Result<(), LangGraphError> {
        if nodes.is_empty() {
            return Err(LangGraphError::GraphError(
                "Graph must contain at least one node".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_node_names_not_empty(
        nodes: &HashMap<String, Box<dyn AgentNode<S>>>,
    ) -> Result<(), LangGraphError> {
        for name in nodes.keys() {
            if name.is_empty() {
                return Err(LangGraphError::GraphError(
                    "Node name cannot be empty".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn validate_start_end_different(
        start_nodes: &HashSet<String>,
        end_node: &str,
    ) -> Result<(), LangGraphError> {
        for start in start_nodes {
            if start == end_node {
                return Err(LangGraphError::GraphError(
                    "Start node and end node cannot be the same".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn validate_start_has_outgoing_edges(
        start_nodes: &HashSet<String>,
        edges: &HashMap<String, HashSet<String>>,
        conditional_edges: &HashMap<String, Vec<RouterFn<S>>>,
    ) -> Result<(), LangGraphError> {
        for start_node in start_nodes {
            let start_has_edge =
                edges.contains_key(start_node) || conditional_edges.contains_key(start_node);
            if !start_has_edge {
                return Err(LangGraphError::GraphError(format!(
                    "Start node '{}' must have at least one outgoing edge",
                    start_node
                )));
            }
        }
        Ok(())
    }

    fn validate_static_edges_valid(
        edges: &HashMap<String, HashSet<String>>,
        nodes: &HashMap<String, Box<dyn AgentNode<S>>>,
        start_nodes: &HashSet<String>,
        end_node: &str,
    ) -> Result<(), LangGraphError> {
        for (from, targets) in edges {
            if !start_nodes.contains(from) && from != end_node && !nodes.contains_key(from) {
                return Err(LangGraphError::GraphError(format!(
                    "Static edge source '{}' is not a registered node",
                    from
                )));
            }
            for target in targets {
                if !start_nodes.contains(target)
                    && target != end_node
                    && !nodes.contains_key(target)
                {
                    return Err(LangGraphError::GraphError(format!(
                        "Static edge target '{}' (from '{}') is not a registered node",
                        target, from
                    )));
                }
            }
        }
        Ok(())
    }

    fn validate_conditional_edges_valid(
        conditional_edges: &HashMap<String, Vec<RouterFn<S>>>,
        nodes: &HashMap<String, Box<dyn AgentNode<S>>>,
        start_nodes: &HashSet<String>,
    ) -> Result<(), LangGraphError> {
        for from in conditional_edges.keys() {
            if !start_nodes.contains(from) && !nodes.contains_key(from) {
                return Err(LangGraphError::GraphError(format!(
                    "Conditional edge source '{}' is not a registered node",
                    from
                )));
            }
        }
        Ok(())
    }

    fn validate_no_mixed_edge_types(
        edges: &HashMap<String, HashSet<String>>,
        conditional_edges: &HashMap<String, Vec<RouterFn<S>>>,
    ) -> Result<(), LangGraphError> {
        for from in edges.keys() {
            if conditional_edges.contains_key(from) {
                return Err(LangGraphError::GraphError(format!(
                    "Node '{}' cannot have both static edges and conditional edges",
                    from
                )));
            }
        }
        Ok(())
    }
}
