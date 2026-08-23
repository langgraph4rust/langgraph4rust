//! Workflow state management.
//!
//! This module defines the [`AgentState`] trait — the contract for any state
//! backend used by the engine — together with [`DefaultMemoryState`], the
//! built-in in-memory implementation.
//!
//! State is the shared, mutable context passed to every [`AgentNode`](crate::AgentNode).
//! Values are stored as JSON internally but accessed with full type safety via
//! serde: [`AgentState::get`] deserializes into a caller-chosen type and
//! [`AgentState::set`] serializes any `Serialize` value.
//!
//! # Concurrency
//!
//! Because parallel nodes may access the state simultaneously, implementations
//! must be `Send + Sync`. [`DefaultMemoryState`] achieves this with an internal
//! `RwLock`-guarded map.

use crate::core::error::LangGraphError;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, from_value, to_value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Trait defining the interface for workflow state management.
///
/// Implement this trait to create custom state backends that can be used with
/// the workflow engine. The state is shared across all nodes in a graph and
/// persists throughout the execution lifecycle.
///
/// # Type Safety
///
/// The trait uses Rust's type system to ensure type-safe serialization and
/// deserialization. Values are stored as JSON internally but accessed with
/// full compile-time type checking.
///
/// # Concurrency Requirements
///
/// State implementations must be thread-safe (`Send` + `Sync`) since they may
/// be accessed concurrently by parallel nodes.
///
/// # Example - Custom State Implementation
///
/// ```rust
/// use langgraph4rust::*;
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use tokio::sync::RwLock;
///
/// #[derive(Clone)]
/// struct PersistentState {
///     data: Arc<RwLock<HashMap<String, String>>>,
/// }
///
/// impl PersistentState {
///     pub fn new() -> Self {
///         Self {
///             data: Arc::new(RwLock::new(HashMap::new())),
///         }
///     }
/// }
///
/// #[async_trait::async_trait]
/// impl AgentState for PersistentState {
///     async fn get<T: serde::de::DeserializeOwned + Send + Sync>(&self, key: &str) -> Result<Option<T>, LangGraphError> {
///         let data = self.data.read().await;
///         match data.get(key) {
///             Some(value) => Ok(serde_json::from_str(value).ok()),
///             None => Ok(None),
///         }
///     }
///
///     async fn set<T: serde::Serialize + Send + Sync>(&self, key: &str, value: T) -> Result<bool, LangGraphError> {
///         let json = serde_json::to_string(&value)
///             .map_err(|e| LangGraphError::StateError(e.to_string()))?;
///         let mut data = self.data.write().await;
///         data.insert(key.to_string(), json);
///         Ok(true)
///     }
///
///     async fn snapshot(&self, _step: usize, _node_keys: Vec<String>) -> Result<(), ()> {
///         Ok(())
///     }
/// }
/// ```
#[async_trait::async_trait]
pub trait AgentState {
    /// Retrieve a value from the state by key.
    ///
    /// Attempts to deserialize the stored JSON value into the requested type `T`.
    ///
    /// # Type Parameters
    ///
    /// - `T`: The target type to deserialize into. Must implement:
    ///   - `DeserializeOwned`: Can be deserialized from owned data
    ///   - `Send`: Safe to transfer between threads
    ///   - `Sync`: Safe to share between threads
    ///
    /// # Arguments
    ///
    /// * `key` - The string key to look up in the state
    ///
    /// # Returns
    ///
    /// - `Ok(Some(T))` if the key exists and deserialization succeeds
    /// - `Ok(None)` if the key does not exist
    /// - `Err(LangGraphError)` if deserialization fails or an I/O error occurs
    ///
    /// # Errors
    ///
    /// Returns [`LangGraphError::StateError`] when:
    /// - The stored value cannot be deserialized into type `T`
    /// - An internal storage error occurs
    ///
    /// # Example
    ///
    /// ```rust
    /// use langgraph4rust::*;
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), LangGraphError> {
    ///     let state = Arc::new(DefaultMemoryState::new());
    ///
    ///     // Get a string value
    ///     let name: Option<String> = state.get("user_name").await?;
    ///
    ///     // Get an integer value
    ///     let count: Option<i32> = state.get("counter").await?;
    ///
    ///     // Get a complex type (must implement Deserialize)
    ///     let data: Option<Vec<String>> = state.get("items").await?;
    ///     Ok(())
    /// }
    /// ```
    async fn get<T: DeserializeOwned + Send + Sync>(
        &self,
        key: &str,
    ) -> Result<Option<T>, LangGraphError>;

    /// Store a value in the state under the given key.
    ///
    /// Serializes the value to JSON and stores it. If the key already exists,
    /// its value will be overwritten.
    ///
    /// # Type Parameters
    ///
    /// - `T`: The type of value to store. Must implement:
    ///   - `Serialize`: Can be serialized to JSON
    ///   - `Send`: Safe to transfer between threads
    ///   - `Sync`: Safe to share between threads
    ///
    /// # Arguments
    ///
    /// * `key` - The string key to associate with the value
    /// * `value` - The value to store (will be serialized to JSON)
    ///
    /// # Returns
    ///
    /// - `Ok(true)` on successful storage
    /// - `Err(LangGraphError)` if serialization fails or an I/O error occurs
    ///
    /// # Errors
    ///
    /// Returns [`LangGraphError::StateError`] when:
    /// - The value cannot be serialized to JSON
    /// - An internal storage error occurs
    ///
    /// # Example
    ///
    /// ```rust
    /// use langgraph4rust::*;
    /// use std::collections::HashMap;
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), LangGraphError> {
    ///     let state = Arc::new(DefaultMemoryState::new());
    ///
    ///     // Store primitive types
    ///     state.set("name", "Alice").await?;
    ///     state.set("age", 30).await?;
    ///     state.set("active", true).await?;
    ///
    ///     // Store complex types
    ///     state.set("tags", vec!["rust", "workflow"]).await?;
    ///     state.set("metadata", HashMap::from([("version", "1.0")])).await?;
    ///     Ok(())
    /// }
    /// ```
    async fn set<T: Serialize + Send + Sync>(
        &self,
        key: &str,
        value: T,
    ) -> Result<bool, LangGraphError>;


    /// Save a snapshot of a value under the given key.
    ///
    /// Provides a hook for state backends to persist checkpoints at critical
    /// points during workflow execution. The default implementation is a no-op;
    /// custom backends can override this to write the state to a database, file
    /// system, or other durable storage.
    ///
    /// Unlike [`set`](AgentState::set), `snapshot` does not modify the runtime
    /// state — it is intended solely for checkpointing purposes. The workflow
    /// engine may call this method automatically after each node execution to
    /// create recovery points.
    ///
    /// # Type Parameters
    ///
    /// - `T`: The type of value to snapshot. Must implement:
    ///   - `Serialize`: Can be serialized to JSON
    ///   - `Send`: Safe to transfer between threads
    ///   - `Sync`: Safe to share between threads
    ///
    /// # Arguments
    ///
    /// * `key` - The string key identifying the value in the state
    /// * `value` - The value to snapshot (will be serialized to JSON)
    ///
    /// # Returns
    ///
    /// - `Ok(())` on successful snapshot
    /// - `Err(LangGraphError)` if serialization or persistence fails
    ///
    /// # Errors
    ///
    /// Returns [`LangGraphError::StateError`] when:
    /// - The value cannot be serialized to JSON
    /// - An internal storage or persistence error occurs
    async fn snapshot(
        &self,
        step: usize,
        node_keys: Vec<String>,
    ) -> Result<(), ()>;
}

/// Default in-memory implementation of [`AgentState`] using JSON storage.
///
/// This is the standard state backend that comes with langgraph4rust. It stores
/// all values as JSON in memory using a `HashMap` protected by a `RwLock`.
///
/// # Features
///
/// - **Thread-safe**: Uses `RwLock` for concurrent access
/// - **Type-safe**: Full compile-time type checking via serde
/// - **JSON-based**: All values serialized as JSON, enabling flexible schemas
/// - **Zero-config**: Works out of the box, no setup required
///
/// # Thread Safety
///
/// Multiple reads can happen concurrently. Writes are exclusive but don't block
/// readers waiting for other readers to finish.
///
/// # Performance Characteristics
///
/// - **Get operations**: O(1) average case for lookup + deserialization cost
/// - **Set operations**: O(1) average case for insertion + serialization cost
/// - **Memory**: Stores all data in RAM; not suitable for very large datasets
///
/// # Use Cases
///
/// - Development and testing
/// - Short-lived workflows
/// - Workflows with moderate data sizes (< 100MB recommended)
/// - When simplicity is preferred over persistence
///
/// For production use cases requiring persistence or larger capacity,
/// consider implementing [`AgentState`] with a database backend.
///
/// # Example
///
/// ```rust
/// use langgraph4rust::*;
/// use std::collections::HashMap;
/// use std::sync::Arc;
///
/// #[tokio::main]
/// async fn main() -> Result<(), LangGraphError> {
///     let state = Arc::new(DefaultMemoryState::new());
///
///     // Store some initial data
///     state.set("counter", 0).await?;
///     state.set("config", HashMap::from([("debug", true)])).await?;
///
///     // Retrieve data
///     let counter: i32 = state.get("counter").await?.unwrap();
///     println!("Initial counter: {}", counter);
///
///     Ok(())
/// }
/// ```
pub struct DefaultMemoryState {
    memory: Arc<RwLock<HashMap<String, Value>>>,
}

impl DefaultMemoryState {
    /// Create a new empty state instance.
    ///
    /// Initializes an empty in-memory state ready for use. No allocation
    /// happens until the first value is stored.
    ///
    /// # Returns
    ///
    /// A fresh `DefaultMemoryState` instance with no data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use langgraph4rust::DefaultMemoryState;
    ///
    /// let state = DefaultMemoryState::new();
    /// // Ready to use - starts empty
    /// ```
    pub fn new() -> Self {
        DefaultMemoryState {
            memory: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for DefaultMemoryState {
    /// Creates a new default instance (same as [`DefaultMemoryState::new()`]).
    ///
    /// This allows using `DefaultMemoryState::default()` or `DefaultMemoryState::default()`
    /// wherever a new state is needed.
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl AgentState for DefaultMemoryState {
    async fn get<T: DeserializeOwned + Send + Sync>(
        &self,
        key: &str,
    ) -> Result<Option<T>, LangGraphError> {
        let memory = self.memory.read().await;

        match memory.get(key) {
            Some(value) => {
                let result = from_value(value.clone()).map_err(|e| {
                    LangGraphError::StateError(format!("Deserialization error: {}", e))
                })?;
                Ok(Some(result))
            }
            None => Ok(None),
        }
    }

    async fn set<T: Serialize + Send + Sync>(
        &self,
        key: &str,
        value: T,
    ) -> Result<bool, LangGraphError> {
        let json_value = to_value(value)
            .map_err(|e| LangGraphError::StateError(format!("Serialization error: {}", e)))?;

        let mut memory = self.memory.write().await;
        memory.insert(key.to_string(), json_value);
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