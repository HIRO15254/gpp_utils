use crate::error::{Error, Result};
use crate::experiment::config::FitnessSpec;
use crate::graph_partition::{Graph, PartitionState};
use std::{collections::BTreeMap, sync::Arc};

pub trait VertexFitness: Send + Sync {
    fn values(&self, graph: &Graph, state: &PartitionState) -> Result<Vec<f64>>;
}
pub trait FitnessFactory: Send + Sync {
    fn version(&self) -> &str;
    fn validate(&self, params: &serde_json::Value) -> Result<()>;
    fn create(&self, params: &serde_json::Value) -> Result<Box<dyn VertexFitness>>;
}
#[derive(Clone)]
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
        Ok((0..g.node_count)
            .map(|v| {
                let d = g.degree(v);
                if d == 0 {
                    1.0
                } else {
                    let good = g
                        .neighbors(v)
                        .iter()
                        .filter(|&&u| s.partition()[u] == s.partition()[v])
                        .count();
                    good as f64 / d as f64
                }
            })
            .collect())
    }
}
