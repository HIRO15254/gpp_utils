use crate::error::{Error, Result};
use crate::experiment::config::FitnessSpec;
use crate::graph_partition::{Graph, PartitionState};
use std::{collections::BTreeMap, sync::Arc};

/// A deterministic vertex-ranking function used by EO.
///
/// Return exactly `graph.node_count()` finite values in vertex ID order; lower
/// values are selected as lower fitness. The same graph, state and factory
/// parameters must produce identical bits. Do not consume exploration RNG or
/// mutate external state that affects values. A deterministic local cache is
/// allowed. The engine validates vector length and finiteness before ranking.
pub trait VertexFitness: Send + Sync {
    /// Evaluate every vertex for a state constructed from this graph.
    fn values(&self, graph: &Graph, state: &PartitionState) -> Result<Vec<f64>>;
}
/// Validates parameters and constructs one fitness evaluator per job.
///
/// Give each semantic definition an immutable version string. Changing returned
/// values, parameter interpretation or defaults requires a new version; stored
/// experiments with another version must not be silently reused.
pub trait FitnessFactory: Send + Sync {
    /// Semantic version stored in the experiment and included in condition IDs.
    fn version(&self) -> &str;
    /// Reject unknown, invalid or noncanonical parameters without side effects.
    /// This method validates rather than normalizes the input JSON.
    fn validate(&self, params: &serde_json::Value) -> Result<()>;
    /// Construct a job-local evaluator after successful parameter validation.
    fn create(&self, params: &serde_json::Value) -> Result<Box<dyn VertexFitness>>;
}
#[derive(Clone)]
/// Named fitness definitions used consistently for compile, run and resume.
///
/// The default registry contains only `default` (`good_edge_fraction-v1`).
/// Custom CLI definitions require rebuilding a caller that supplies this
/// registry. See `examples/custom_fitness.rs` for a full persisted lifecycle.
pub struct FitnessRegistry {
    entries: BTreeMap<String, Arc<dyn FitnessFactory>>,
}
impl Default for FitnessRegistry {
    fn default() -> Self {
        let mut x = Self {
            entries: BTreeMap::new(),
        };
        x.register("default", Arc::new(DefaultFactory));
        x
    }
}
impl FitnessRegistry {
    pub fn default_registry() -> Self {
        Self::default()
    }
    /// Register a definition, returning the previous factory if the name exists.
    ///
    /// Replacement is explicit caller policy: check the returned `Option` if
    /// duplicates should be rejected. Do not replace a definition with different
    /// behavior under the same version. The built-in `default` name is reserved
    /// by the compiler's pinned definition; use a new name for custom fitness.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        factory: Arc<dyn FitnessFactory>,
    ) -> Option<Arc<dyn FitnessFactory>> {
        self.entries.insert(name.into(), factory)
    }
    pub fn versions(&self) -> BTreeMap<String, String> {
        self.entries
            .iter()
            .map(|(k, v)| (k.clone(), v.version().to_owned()))
            .collect()
    }
    pub fn validate(&self, spec: &FitnessSpec) -> Result<()> {
        self.entries
            .get(&spec.kind)
            .ok_or_else(|| Error::msg(format!("unknown fitness: {}", spec.kind)))?
            .validate(&spec.params)
    }
    pub fn create(&self, spec: &FitnessSpec) -> Result<Box<dyn VertexFitness>> {
        let f = self
            .entries
            .get(&spec.kind)
            .ok_or_else(|| Error::msg(format!("unknown fitness: {}", spec.kind)))?;
        f.validate(&spec.params)?;
        f.create(&spec.params)
    }
}
impl std::fmt::Debug for FitnessRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FitnessRegistry")
            .field("versions", &self.versions())
            .finish()
    }
}
struct DefaultFactory;
impl FitnessFactory for DefaultFactory {
    fn version(&self) -> &str {
        "good_edge_fraction-v1"
    }
    fn validate(&self, p: &serde_json::Value) -> Result<()> {
        if p.as_object().is_some_and(|x| x.is_empty()) || p.is_null() {
            Ok(())
        } else {
            Err(Error::msg("default fitness params must be empty"))
        }
    }
    fn create(&self, _: &serde_json::Value) -> Result<Box<dyn VertexFitness>> {
        Ok(Box::new(DefaultFitness))
    }
}
struct DefaultFitness;
impl VertexFitness for DefaultFitness {
    fn values(&self, g: &Graph, s: &PartitionState) -> Result<Vec<f64>> {
        Ok((0..g.node_count())
            .map(|v| {
                let d = g.degree(v);
                if d == 0 {
                    1.0
                } else {
                    let good = d as i64 - s.cuts_at()[v];
                    good as f64 / d as f64
                }
            })
            .collect())
    }
}
