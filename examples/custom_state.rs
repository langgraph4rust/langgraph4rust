//! # Custom State Implementation Example
//!
//! Demonstrates how to implement a custom [`AgentState`] backend
//! for specialized use cases like logging, persistence, or validation.
//!
//! ## Run this example:
//!
//! ```bash
//! cargo run --example custom_state
//! ```

use async_trait::async_trait;
use langgraph4rust::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use serde::Serialize;
// ============================================================================
// Custom State Implementation with Audit Logging
// ============================================================================

/// A custom state implementation that logs all operations
#[derive(Clone)]
struct AuditedState {
    data: Arc<tokio::sync::RwLock<HashMap<String, serde_json::Value>>>,
    audit_log: Arc<std::sync::Mutex<Vec<String>>>,
}

impl AuditedState {
    pub fn new() -> Self {
        AuditedState {
            data: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            audit_log: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Retrieve the complete audit log
    pub fn get_audit_log(&self) -> Vec<String> {
        self.audit_log.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl AgentState for AuditedState {
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

                // Log the read operation (use debug representation)
                let log_entry = format!("READ {} = <value of type>", key);
                self.audit_log.lock().unwrap().push(log_entry);

                Ok(Some(result))
            }
            None => {
                self.audit_log
                    .lock()
                    .unwrap()
                    .push(format!("READ {} = <not found>", key));
                Ok(None)
            }
        }
    }

    async fn set<T: serde::Serialize + Send + Sync>(
        &self,
        key: &str,
        value: T,
    ) -> Result<bool, LangGraphError> {
        let json_value = serde_json::to_value(value)
            .map_err(|e| LangGraphError::StateError(format!("Serialization error: {}", e)))?;

        // Log the write operation
        let log_entry = format!("SET {} = {}", key, json_value);
        self.audit_log.lock().unwrap().push(log_entry);

        let mut data = self.data.write().await;
        data.insert(key.to_string(), json_value);
        Ok(true)
    }

    async fn snapshot(
        &self,
        _: usize,
        _: Vec<String>,
    ) -> Result<(), ()>{
        Ok(())
    }
}

// ============================================================================
// Node Definitions
// ============================================================================

/// Node that performs some operations on the audited state
#[derive(Clone)]
struct DataProcessor;

#[async_trait]
impl AgentNode<AuditedState> for DataProcessor {
    async fn apply(&self, state: Arc<AuditedState>) -> Result<(), LangGraphError> {
        println!("📝 Processing data...");

        // Write some values
        state.set("user", "Alice").await?;
        state.set("score", 95).await?;
        state.set("active", true).await?;

        // Read and modify
        let score: i32 = state.get("score").await?.unwrap_or(0);
        state.set("doubled_score", score * 2).await?;

        println!("   ✅ Data processing complete");
        Ok(())
    }
}

// ============================================================================
// Main Execution
// ============================================================================

#[tokio::main]
async fn main() -> Result<(), LangGraphError> {
    println!("=== Custom State Implementation Example ===\n");

    // Build workflow using custom state type
    let mut builder = StateGraphBuilder::<AuditedState>::new();
    builder.add_node("process", Box::new(DataProcessor));

    builder.add_edge(START_NODE, HashSet::from(["process".to_string()]));
    builder.add_edge("process", HashSet::from([END_NODE.to_string()]));

    let graph = builder.compile()?;
    let state = Arc::new(AuditedState::new());

    println!("🔨 Building workflow with custom AuditedState...\n");
    graph.invoke(state.clone()).await?;

    println!("\n📋 Complete Audit Log:");
    println!("{}", "─".repeat(50));

    let log = state.get_audit_log();
    for (i, entry) in log.iter().enumerate() {
        println!("{:3}. {}", i + 1, entry);
    }

    println!("{}", "─".repeat(50));
    println!("\n✅ Total operations logged: {}", log.len());

    // Verify the data is correct
    println!("\n📊 Final State Values:");
    if let Some(user) = state.get::<String>("user").await? {
        println!("   user: {}", user);
    }
    if let Some(score) = state.get::<i32>("score").await? {
        println!("   score: {}", score);
    }
    if let Some(doubled) = state.get::<i32>("doubled_score").await? {
        println!("   doubled_score: {}", doubled);
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
