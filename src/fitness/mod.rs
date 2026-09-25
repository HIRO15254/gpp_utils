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

/// Registered name and semantic version of every built-in definition.
///
/// These names are reserved: the compiler pins their versions, so replacing one
/// through [`FitnessRegistry::register`] makes experiments using it fail to
/// compile.
pub const BUILTIN_FITNESSES: [(&str, &str); 3] = [
    ("default", "good_edge_fraction-v1"),
    ("multiplicative", "multiplicative-v1"),
    ("additive", "additive-v1"),
];

#[derive(Clone)]
struct Entry {
    factory: Arc<dyn FitnessFactory>,
    /// Set only for the built-in definitions installed by [`FitnessRegistry::default`].
    /// A registered replacement always clears it, so EO never applies the
    /// built-in incremental ranking to a caller-supplied definition.
    builtin: Option<BuiltinKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuiltinKind {
    Default,
    Multiplicative,
    Additive,
}

#[derive(Clone)]
/// Named fitness definitions used consistently for compile, run and resume.
///
/// The default registry contains the built-in `default` (`good_edge_fraction-v1`),
/// `multiplicative` (`multiplicative-v1`, parameter `alpha`) and `additive`
/// (`additive-v1`, parameter `beta`) definitions. Custom CLI definitions require
/// rebuilding a caller that supplies this registry. See
/// `examples/custom_fitness.rs` for a full persisted lifecycle.
pub struct FitnessRegistry {
    entries: BTreeMap<String, Entry>,
}
impl Default for FitnessRegistry {
    fn default() -> Self {
        let builtins: [(&str, Arc<dyn FitnessFactory>, BuiltinKind); 3] = [
            ("default", Arc::new(DefaultFactory), BuiltinKind::Default),
            (
                "multiplicative",
                Arc::new(MultiplicativeFactory),
                BuiltinKind::Multiplicative,
            ),
            ("additive", Arc::new(AdditiveFactory), BuiltinKind::Additive),
        ];
        Self {
            entries: builtins
                .into_iter()
                .map(|(name, factory, kind)| {
                    (
                        name.to_owned(),
                        Entry {
                            factory,
                            builtin: Some(kind),
                        },
                    )
                })
                .collect(),
        }
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
    /// behavior under the same version. The built-in names in
    /// [`BUILTIN_FITNESSES`] are reserved by the compiler's pinned definitions;
    /// use a new name for custom fitness.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        factory: Arc<dyn FitnessFactory>,
    ) -> Option<Arc<dyn FitnessFactory>> {
        self.entries
            .insert(
                name.into(),
                Entry {
                    factory,
                    builtin: None,
                },
            )
            .map(|entry| entry.factory)
    }
    pub fn versions(&self) -> BTreeMap<String, String> {
        self.entries
            .iter()
            .map(|(k, v)| (k.clone(), v.factory.version().to_owned()))
            .collect()
    }
    fn entry(&self, spec: &FitnessSpec) -> Result<&Entry> {
        self.entries
            .get(&spec.kind)
            .ok_or_else(|| Error::msg(format!("unknown fitness: {}", spec.kind)))
    }
    pub fn validate(&self, spec: &FitnessSpec) -> Result<()> {
        self.entry(spec)?.factory.validate(&spec.params)
    }
    pub fn create(&self, spec: &FitnessSpec) -> Result<Box<dyn VertexFitness>> {
        let entry = self.entry(spec)?;
        entry.factory.validate(&spec.params)?;
        entry.factory.create(&spec.params)
    }
    /// Resolve a definition for the EO engine.
    ///
    /// Built-in definitions installed by [`Self::default`] are returned in their
    /// structured form so EO can rank them incrementally. Every other definition
    /// is evaluated through [`VertexFitness::values`].
    pub(crate) fn create_engine_fitness(&self, spec: &FitnessSpec) -> Result<EngineFitness> {
        let entry = self.entry(spec)?;
        entry.factory.validate(&spec.params)?;
        Ok(match entry.builtin {
            Some(kind) => EngineFitness::Builtin(BuiltinFitness::parse(kind, &spec.params)?),
            None => EngineFitness::Custom(entry.factory.create(&spec.params)?),
        })
    }
}
impl std::fmt::Debug for FitnessRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FitnessRegistry")
            .field("versions", &self.versions())
            .finish()
    }
}

/// How the EO engine evaluates a resolved fitness definition.
pub(crate) enum EngineFitness {
    /// A built-in definition, ranked incrementally from `(degree, cuts, side)`.
    Builtin(BuiltinFitness),
    /// Any other definition, evaluated for every vertex at every step.
    Custom(Box<dyn VertexFitness>),
}

/// A built-in fitness `lambda = f(lambda0, majority)`.
///
/// `lambda0` is [`lambda0`] (the `default` definition). A vertex is majority
/// when its group is strictly larger than the other group; with equal sizes no
/// vertex is majority. [`Self::lambda`] is the single definition of every
/// built-in value, used by both [`VertexFitness::values`] and the EO index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum BuiltinFitness {
    /// `lambda0`.
    Default,
    /// `lambda0 * alpha` for majority vertices, `lambda0 * 1.0` otherwise; `0 <= alpha <= 1`.
    Multiplicative { alpha: f64 },
    /// `beta * lambda0 + 0.0` for majority vertices, `beta * lambda0 + 1.0` otherwise; `beta >= 0`.
    Additive { beta: f64 },
}
impl BuiltinFitness {
    fn parse(kind: BuiltinKind, params: &serde_json::Value) -> Result<Self> {
        Ok(match kind {
            BuiltinKind::Default => Self::Default,
            BuiltinKind::Multiplicative => Self::Multiplicative {
                alpha: multiplicative_alpha(params)?,
            },
            BuiltinKind::Additive => Self::Additive {
                beta: additive_beta(params)?,
            },
        })
    }
    /// The fitness of a vertex with default value `lambda0` and majority flag.
    #[inline]
    pub(crate) fn lambda(self, lambda0: f64, is_majority: bool) -> f64 {
        match self {
            Self::Default => lambda0,
            Self::Multiplicative { alpha } => lambda0 * if is_majority { alpha } else { 1.0 },
            Self::Additive { beta } => beta * lambda0 + if is_majority { 0.0 } else { 1.0 },
        }
    }
    /// Whether majority and minority vertices can receive different values.
    ///
    /// When false, the value depends only on `lambda0`, so every group-size
    /// relation ranks vertices identically.
    pub(crate) fn depends_on_majority(self) -> bool {
        match self {
            Self::Default => false,
            Self::Multiplicative { alpha } => alpha != 1.0,
            Self::Additive { .. } => true,
        }
    }
}

/// The `default` fitness `g/deg = (degree - cuts) / degree`; isolated vertices are 1.
#[inline]
pub(crate) fn lambda0(degree: usize, cuts: i64) -> f64 {
    if degree == 0 {
        1.0
    } else {
        let good = degree as i64 - cuts;
        good as f64 / degree as f64
    }
}

/// Whether a vertex on `side` (`true` = group A) belongs to the strictly larger group.
#[inline]
pub(crate) fn is_majority(side: bool, size_a: usize, size_b: usize) -> bool {
    if side {
        size_a > size_b
    } else {
        size_b > size_a
    }
}

fn builtin_values(kind: BuiltinFitness, g: &Graph, s: &PartitionState) -> Vec<f64> {
    let (size_a, size_b) = (s.size_a(), s.size_b());
    (0..g.node_count())
        .map(|v| {
            let side = s.partition()[v];
            kind.lambda(
                lambda0(g.degree(v), s.cuts_at()[v]),
                is_majority(side, size_a, size_b),
            )
        })
        .collect()
}

/// Read the single floating-point parameter `key`.
///
/// Integer JSON numbers are rejected rather than converted, because the
/// canonical condition hash distinguishes `1` from `1.0`.
fn float_param(params: &serde_json::Value, key: &str, kind: &str) -> Result<f64> {
    let object = params
        .as_object()
        .filter(|object| object.len() == 1 && object.contains_key(key))
        .ok_or_else(|| Error::msg(format!("{kind} fitness params must be exactly {{ {key} }}")))?;
    let value = match &object[key] {
        serde_json::Value::Number(number) if number.is_f64() => number.as_f64(),
        _ => None,
    }
    .ok_or_else(|| {
        Error::msg(format!(
            "{kind} fitness {key} must be a floating-point number such as 1.0"
        ))
    })?;
    if !value.is_finite() {
        return Err(Error::msg(format!("{kind} fitness {key} must be finite")));
    }
    Ok(value)
}
fn multiplicative_alpha(params: &serde_json::Value) -> Result<f64> {
    let alpha = float_param(params, "alpha", "multiplicative")?;
    if alpha.is_sign_negative() || alpha > 1.0 {
        return Err(Error::msg(
            "multiplicative fitness alpha must satisfy 0.0 <= alpha <= 1.0",
        ));
    }
    Ok(alpha)
}
fn additive_beta(params: &serde_json::Value) -> Result<f64> {
    let beta = float_param(params, "beta", "additive")?;
    if beta.is_sign_negative() {
        return Err(Error::msg("additive fitness beta must be non-negative"));
    }
    Ok(beta)
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
        Ok(Box::new(BuiltinVertexFitness(BuiltinFitness::Default)))
    }
}
struct MultiplicativeFactory;
impl FitnessFactory for MultiplicativeFactory {
    fn version(&self) -> &str {
        "multiplicative-v1"
    }
    fn validate(&self, p: &serde_json::Value) -> Result<()> {
        multiplicative_alpha(p).map(|_| ())
    }
    fn create(&self, p: &serde_json::Value) -> Result<Box<dyn VertexFitness>> {
        Ok(Box::new(BuiltinVertexFitness(
            BuiltinFitness::Multiplicative {
                alpha: multiplicative_alpha(p)?,
            },
        )))
    }
}
struct AdditiveFactory;
impl FitnessFactory for AdditiveFactory {
    fn version(&self) -> &str {
        "additive-v1"
    }
    fn validate(&self, p: &serde_json::Value) -> Result<()> {
        additive_beta(p).map(|_| ())
    }
    fn create(&self, p: &serde_json::Value) -> Result<Box<dyn VertexFitness>> {
        Ok(Box::new(BuiltinVertexFitness(BuiltinFitness::Additive {
            beta: additive_beta(p)?,
        })))
    }
}
struct BuiltinVertexFitness(BuiltinFitness);
impl VertexFitness for BuiltinVertexFitness {
    fn values(&self, g: &Graph, s: &PartitionState) -> Result<Vec<f64>> {
        Ok(builtin_values(self.0, g, s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(kind: &str, params: serde_json::Value) -> FitnessSpec {
        FitnessSpec {
            kind: kind.into(),
            params,
        }
    }

    #[test]
    fn builtin_versions_are_registered() {
        let versions = FitnessRegistry::default().versions();
        for (name, version) in BUILTIN_FITNESSES {
            assert_eq!(versions.get(name).map(String::as_str), Some(version));
        }
        assert_eq!(versions.len(), BUILTIN_FITNESSES.len());
    }

    #[test]
    fn parameters_are_validated_without_normalization() {
        let r = FitnessRegistry::default();
        for ok in [
            spec("multiplicative", json!({"alpha": 0.0})),
            spec("multiplicative", json!({"alpha": 0.5})),
            spec("multiplicative", json!({"alpha": 1.0})),
            spec("additive", json!({"beta": 0.0})),
            spec("additive", json!({"beta": 32.0})),
        ] {
            r.validate(&ok).unwrap();
        }
        for bad in [
            spec("multiplicative", json!({"alpha": 1})),
            spec("multiplicative", json!({"alpha": -0.0})),
            spec("multiplicative", json!({"alpha": 1.5})),
            spec("multiplicative", json!({"alpha": -0.1})),
            spec("multiplicative", json!({})),
            spec("multiplicative", json!({"alpha": 0.5, "beta": 1.0})),
            spec("multiplicative", json!({"beta": 0.5})),
            spec("multiplicative", serde_json::Value::Null),
            spec("additive", json!({"beta": 3})),
            spec("additive", json!({"beta": -0.0})),
            spec("additive", json!({"beta": -1.0})),
            spec("additive", json!({"beta": "3.0"})),
            spec("default", json!({"alpha": 0.5})),
        ] {
            assert!(r.validate(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn builtin_values_follow_the_majority_rule() {
        // Path 0-1-2-3 plus isolated vertex 4; groups A={0,1,2}, B={3,4}.
        let g = Graph::from_edges(5, vec![[0, 1], [1, 2], [2, 3]]).unwrap();
        let s = PartitionState::new(&g, vec![true, true, true, false, false]).unwrap();
        let l0 = [1.0, 1.0, 0.5, 0.0, 1.0];
        let r = FitnessRegistry::default();
        let values = |kind: &str, params| r.create(&spec(kind, params)).unwrap().values(&g, &s);
        assert_eq!(values("default", json!({})).unwrap(), l0);
        // A is the majority, so only vertices 0..=2 are scaled.
        assert_eq!(
            values("multiplicative", json!({"alpha": 0.5})).unwrap(),
            [0.5, 0.5, 0.25, 0.0, 1.0]
        );
        assert_eq!(
            values("additive", json!({"beta": 2.0})).unwrap(),
            [2.0, 2.0, 1.0, 1.0, 3.0]
        );
        // Equal group sizes: nobody is majority.
        let even = Graph::from_edges(4, vec![[0, 1], [1, 2], [2, 3]]).unwrap();
        let s = PartitionState::new(&even, vec![true, true, false, false]).unwrap();
        let fitness = r
            .create(&spec("multiplicative", json!({"alpha": 0.0})))
            .unwrap();
        assert_eq!(
            fitness.values(&even, &s).unwrap(),
            [1.0, 0.5, 0.5, 1.0],
            "balanced groups have no majority"
        );
    }

    #[test]
    fn only_default_registry_entries_use_the_builtin_engine_path() {
        let mut r = FitnessRegistry::default();
        assert!(matches!(
            r.create_engine_fitness(&spec("additive", json!({"beta": 3.0})))
                .unwrap(),
            EngineFitness::Builtin(BuiltinFitness::Additive { beta }) if beta == 3.0
        ));
        r.register("additive", Arc::new(AdditiveFactory));
        assert!(matches!(
            r.create_engine_fitness(&spec("additive", json!({"beta": 3.0})))
                .unwrap(),
            EngineFitness::Custom(_)
        ));
    }

    #[test]
    fn majority_independence_is_detected() {
        assert!(!BuiltinFitness::Default.depends_on_majority());
        assert!(!BuiltinFitness::Multiplicative { alpha: 1.0 }.depends_on_majority());
        assert!(BuiltinFitness::Multiplicative { alpha: 0.5 }.depends_on_majority());
        assert!(BuiltinFitness::Additive { beta: 0.0 }.depends_on_majority());
    }
}
