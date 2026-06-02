//! SA 実行条件の設定。
//!
//! - 温度は Θ = log_10(T) で指定する。Θ = None で T = 0（受理しない）。
//! - イテレーション数は 10^N で指定する（N は整数）。
//! - スムージング戦略を選択できる。

use serde::{Deserialize, Serialize};

/// スムージング指定。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SmoothingSpec {
    None,
    /// 決定論的 K-近傍平均。
    KAverage(usize),
    /// 確率的 K-近傍平均（距離 2 フォールバックあり）。
    RandomKAverage(usize),
    /// 重み付き平均（w × 全近傍平均 + (1-w) × current）。`w` は重み（0〜1）。
    WeightedAverage(f64),
}

/// 重み値を id 文字列用に整形する（小数点 `.` は `p` に置換）。
fn fmt_weight(w: f64) -> String {
    if w.fract().abs() < 1e-9 {
        format!("{}", w as i64)
    } else {
        format!("{}", w).replace('.', "p")
    }
}

/// τ 値を id 文字列用に整形する（小数点 `.` は `p` に置換）。例: 1.4 → `1p4`、2.0 → `2`。
fn fmt_tau(t: f64) -> String {
    if t.fract().abs() < 1e-9 {
        format!("{}", t as i64)
    } else {
        format!("{}", t).replace('.', "p")
    }
}

/// τ-EO の既定の指数 τ（スペック既定値、実用範囲 1.3〜1.6）。
pub const DEFAULT_TAU: f64 = 1.4;

/// ソルバー指定。`Sa` は固定温度メトロポリス（既存）、`Eo` は τ-EO（厳密バランスのスワップ）。
///
/// `RunConfig` に `#[serde(default)]` で埋め込まれるため、`solver` フィールドを持たない
/// 既存の JSON（結果・batch・回帰 baseline）は `Sa` として読み込まれる（後方互換）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SolverSpec {
    /// 固定温度シミュレーテッドアニーリング（温度は `RunConfig::theta`）。
    Sa,
    /// τ-Extremal Optimization（厳密バランスのスワップ版）。`tau` はべき乗則指数（既定 [`DEFAULT_TAU`]）。
    Eo { tau: f64 },
    /// τ-Extremal Optimization（フリップ近傍版）。SA と同一の近傍・目的関数・ベイスン算出を共有し、
    /// バランスはペナルティ項で扱う。適応度は g/deg にバランスペナルティを「悪い辺/良い辺」として
    /// 織り込んだもの。`tau` はべき乗則指数（既定 [`DEFAULT_TAU`]）。
    EoFlip { tau: f64 },
}

impl Default for SolverSpec {
    fn default() -> Self {
        SolverSpec::Sa
    }
}

/// ソルバーの種別（パラメータを持たない）。sweep で「種別 × 複数 τ」を表すのに使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverKind {
    Sa,
    Eo,
    EoFlip,
}

impl Default for SolverKind {
    fn default() -> Self {
        SolverKind::Sa
    }
}

impl SmoothingSpec {
    pub fn label(&self) -> String {
        match self {
            Self::None => "none".into(),
            Self::KAverage(k) => format!("kavg{}", k),
            Self::RandomKAverage(k) => format!("rkavg{}", k),
            Self::WeightedAverage(w) => format!("wavg{}", fmt_weight(*w)),
        }
    }

    pub fn has_smoothing(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// SA 実行条件。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    /// 表示用ラベル。
    pub name: String,
    /// 温度 Θ = log_10(T)。`None` なら T = 0（悪化拒否）。
    pub theta: Option<f64>,
    /// イテレーション数の指数（max_iter = 10^N）。
    pub log10_iterations: u32,
    /// スムージング戦略。
    pub smoothing: SmoothingSpec,
    /// ソルバー（既定 = SA）。`Eo` のときは `theta`・`smoothing` は無視される。
    #[serde(default)]
    pub solver: SolverSpec,
}

impl RunConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            theta: Some(0.0),
            log10_iterations: 4,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::Sa,
        }
    }

    /// 実温度。Θ = None なら 0、それ以外は 10^Θ。
    pub fn temperature(&self) -> f64 {
        match self.theta {
            None => 0.0,
            Some(t) => 10f64.powf(t),
        }
    }

    /// 反復回数。
    pub fn iterations(&self) -> usize {
        let n = self.log10_iterations.min(9) as u32;
        10usize.pow(n)
    }

    /// 一意な識別子（キャッシュキー用）。
    ///
    /// SA の id は従来形式をバイト単位で維持する（既存キャッシュ dir・回帰 baseline 温存）。
    /// EO は theta/smoothing を無視するため、それらを含めない独立した名前空間にする。
    pub fn id(&self) -> String {
        match self.solver {
            SolverSpec::Sa => {
                let theta = match self.theta {
                    None => "T0".to_string(),
                    Some(t) => {
                        if (t.fract()).abs() < 1e-9 {
                            format!("th{:+}", t as i64)
                        } else {
                            format!("th{:+.2}", t).replace('.', "p")
                        }
                    }
                };
                format!("{}_iter{}_{}", theta, self.log10_iterations, self.smoothing.label())
            }
            SolverSpec::Eo { tau } => {
                format!("eo_iter{}_tau{}", self.log10_iterations, fmt_tau(tau))
            }
            SolverSpec::EoFlip { tau } => {
                format!("eoflip_iter{}_tau{}", self.log10_iterations, fmt_tau(tau))
            }
        }
    }
}

/// 平滑化の種別（K 値を持たない）。sweep 指定で「種別 × 複数 K」を表すのに使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmoothingKind {
    None,
    KAverage,
    RandomKAverage,
    WeightedAverage,
}

impl Default for SmoothingKind {
    fn default() -> Self {
        SmoothingKind::None
    }
}

impl SmoothingKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::KAverage => "kavg",
            Self::RandomKAverage => "rkavg",
            Self::WeightedAverage => "wavg",
        }
    }

    /// 整数 K 値を必要とするか（KAverage / RandomKAverage）。
    pub fn uses_k(self) -> bool {
        matches!(self, Self::KAverage | Self::RandomKAverage)
    }

    /// 重み値（0〜1）を必要とするか（WeightedAverage）。
    pub fn uses_weight(self) -> bool {
        matches!(self, Self::WeightedAverage)
    }
}

/// 温度・反復回数・平滑化 K の各軸を複数指定し、その直積で `RunConfig` 群を生成する指定。
///
/// 平滑化は「単一種別 × 複数 K」の形をとる（`smoothing_kind` が `None` のときは
/// `ks` を無視し、平滑化なしの 1 通りとなる）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSweep {
    /// 温度 Θ の候補。`null` は T = 0（貪欲）。EO sweep では不要（省略可）。
    #[serde(default)]
    pub thetas: Vec<Option<f64>>,
    /// 反復回数指数 N（max_iter = 10^N）の候補。
    pub log10_iterations: Vec<u32>,
    /// 平滑化の種別（生成される全 `RunConfig` で共通）。EO sweep では不要（省略可、既定 None）。
    #[serde(default)]
    pub smoothing_kind: SmoothingKind,
    /// 平滑化の K 値候補（`smoothing_kind` が KAverage / RandomKAverage のとき使用）。
    #[serde(default)]
    pub ks: Vec<usize>,
    /// 平滑化の重み候補 0〜1（`smoothing_kind` が WeightedAverage のとき使用）。
    #[serde(default)]
    pub weights: Vec<f64>,
    /// ソルバー種別（生成される全 `RunConfig` で共通、既定 = SA）。
    #[serde(default)]
    pub solver_kind: SolverKind,
    /// τ-EO の指数 τ 候補（`solver_kind` が `Eo` のとき直積軸に使う。空なら [`DEFAULT_TAU`] 1 通り）。
    #[serde(default)]
    pub taus: Vec<f64>,
}

impl ConfigSweep {
    /// `RunConfig` 群を直積展開する。各 `RunConfig` の `name` には一意な `id()` 文字列が入る。
    ///
    /// - `solver_kind = Sa`（既定）: 従来どおり `thetas × log10_iterations × smoothings`。
    /// - `solver_kind = Eo`: theta/smoothing/ks/weights を無視し `log10_iterations × taus`
    ///   （`taus` が空なら `[DEFAULT_TAU]`）を展開する。各 cfg は `theta: None, smoothing: None`。
    pub fn expand(&self) -> Vec<RunConfig> {
        match self.solver_kind {
            SolverKind::Sa => self.expand_sa(),
            SolverKind::Eo => self.expand_eo(false),
            SolverKind::EoFlip => self.expand_eo(true),
        }
    }

    fn expand_sa(&self) -> Vec<RunConfig> {
        let smoothings: Vec<SmoothingSpec> = match self.smoothing_kind {
            SmoothingKind::None => vec![SmoothingSpec::None],
            SmoothingKind::KAverage => {
                self.ks.iter().map(|&k| SmoothingSpec::KAverage(k)).collect()
            }
            SmoothingKind::RandomKAverage => {
                self.ks.iter().map(|&k| SmoothingSpec::RandomKAverage(k)).collect()
            }
            SmoothingKind::WeightedAverage => {
                self.weights.iter().map(|&w| SmoothingSpec::WeightedAverage(w)).collect()
            }
        };
        let mut out = Vec::new();
        for &theta in &self.thetas {
            for &log10_iterations in &self.log10_iterations {
                for &smoothing in &smoothings {
                    let mut cfg = RunConfig {
                        name: String::new(),
                        theta,
                        log10_iterations,
                        smoothing,
                        solver: SolverSpec::Sa,
                    };
                    cfg.name = cfg.id();
                    out.push(cfg);
                }
            }
        }
        out
    }

    fn expand_eo(&self, flip: bool) -> Vec<RunConfig> {
        let taus: Vec<f64> = if self.taus.is_empty() {
            vec![DEFAULT_TAU]
        } else {
            self.taus.clone()
        };
        let mut out = Vec::new();
        for &log10_iterations in &self.log10_iterations {
            for &tau in &taus {
                let solver = if flip {
                    SolverSpec::EoFlip { tau }
                } else {
                    SolverSpec::Eo { tau }
                };
                let mut cfg = RunConfig {
                    name: String::new(),
                    theta: None,
                    log10_iterations,
                    smoothing: SmoothingSpec::None,
                    solver,
                };
                cfg.name = cfg.id();
                out.push(cfg);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature() {
        let mut c = RunConfig::new("c");
        c.theta = Some(0.0);
        assert!((c.temperature() - 1.0).abs() < 1e-12);
        c.theta = Some(1.0);
        assert!((c.temperature() - 10.0).abs() < 1e-12);
        c.theta = None;
        assert_eq!(c.temperature(), 0.0);
    }

    #[test]
    fn test_iterations() {
        let mut c = RunConfig::new("c");
        c.log10_iterations = 3;
        assert_eq!(c.iterations(), 1000);
        c.log10_iterations = 6;
        assert_eq!(c.iterations(), 1_000_000);
    }

    #[test]
    fn test_id_format() {
        let c = RunConfig {
            name: "a".into(),
            theta: Some(0.0),
            log10_iterations: 4,
            smoothing: SmoothingSpec::KAverage(8),
            solver: SolverSpec::Sa,
        };
        assert_eq!(c.id(), "th+0_iter4_kavg8");
        let c0 = RunConfig {
            name: "a".into(),
            theta: None,
            log10_iterations: 5,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::Sa,
        };
        assert_eq!(c0.id(), "T0_iter5_none");
    }

    #[test]
    fn test_id_format_eo() {
        // EO は theta/smoothing を無視し、独立した名前空間の id を持つ。
        let c = RunConfig {
            name: "a".into(),
            theta: Some(0.0),
            log10_iterations: 4,
            smoothing: SmoothingSpec::KAverage(8),
            solver: SolverSpec::Eo { tau: 1.4 },
        };
        assert_eq!(c.id(), "eo_iter4_tau1p4");
        let c2 = RunConfig {
            name: "a".into(),
            theta: None,
            log10_iterations: 6,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::Eo { tau: 2.0 },
        };
        assert_eq!(c2.id(), "eo_iter6_tau2");
        let c3 = RunConfig {
            name: "a".into(),
            theta: Some(0.0),
            log10_iterations: 5,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::EoFlip { tau: 1.4 },
        };
        assert_eq!(c3.id(), "eoflip_iter5_tau1p4");
    }

    #[test]
    fn test_solver_serde_default() {
        // `solver` フィールドを持たない JSON は Sa として読み込まれる（後方互換）。
        let json = r#"{"name":"x","theta":0.0,"log10_iterations":4,"smoothing":"None"}"#;
        let cfg: RunConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(cfg.solver, SolverSpec::Sa);
        assert_eq!(cfg.id(), "th+0_iter4_none");

        // EO は明示的にデシリアライズできる。
        let json_eo =
            r#"{"name":"x","theta":null,"log10_iterations":5,"smoothing":"None","solver":{"Eo":{"tau":1.4}}}"#;
        let cfg_eo: RunConfig = serde_json::from_str(json_eo).expect("deserialize eo");
        assert_eq!(cfg_eo.solver, SolverSpec::Eo { tau: 1.4 });
        assert_eq!(cfg_eo.id(), "eo_iter5_tau1p4");
    }

    #[test]
    fn test_config_sweep_expand() {
        let sweep = ConfigSweep {
            thetas: vec![Some(0.0), None],
            log10_iterations: vec![4, 5],
            smoothing_kind: SmoothingKind::KAverage,
            ks: vec![4, 8],
            weights: vec![],
            solver_kind: SolverKind::Sa,
            taus: vec![],
        };
        // 2 thetas x 2 iters x 2 Ks = 8
        let cfgs = sweep.expand();
        assert_eq!(cfgs.len(), 8);
        // 生成された RunConfig の name は id() と一致する。
        for c in &cfgs {
            assert_eq!(c.name, c.id());
            assert_eq!(c.solver, SolverSpec::Sa);
        }

        // None 種別は ks / weights を無視し、平滑化なしの 1 通りに展開される。
        let none_sweep = ConfigSweep {
            thetas: vec![Some(1.0)],
            log10_iterations: vec![3],
            smoothing_kind: SmoothingKind::None,
            ks: vec![4, 8, 16],
            weights: vec![0.5],
            solver_kind: SolverKind::Sa,
            taus: vec![],
        };
        let none_cfgs = none_sweep.expand();
        assert_eq!(none_cfgs.len(), 1);
        assert_eq!(none_cfgs[0].smoothing, SmoothingSpec::None);

        // WeightedAverage 種別は weights を直積軸に使う（ks は無視）。
        let w_sweep = ConfigSweep {
            thetas: vec![Some(0.0)],
            log10_iterations: vec![4],
            smoothing_kind: SmoothingKind::WeightedAverage,
            ks: vec![4, 8],
            weights: vec![0.25, 0.5, 1.0],
            solver_kind: SolverKind::Sa,
            taus: vec![],
        };
        let w_cfgs = w_sweep.expand();
        assert_eq!(w_cfgs.len(), 3);
        assert_eq!(w_cfgs[0].smoothing, SmoothingSpec::WeightedAverage(0.25));
    }

    #[test]
    fn test_config_sweep_expand_eo() {
        // EO sweep: log10_iterations × taus（theta/smoothing/ks/weights は無視）。
        let sweep = ConfigSweep {
            thetas: vec![Some(0.0), None],
            log10_iterations: vec![4, 5],
            smoothing_kind: SmoothingKind::KAverage,
            ks: vec![4, 8],
            weights: vec![0.5],
            solver_kind: SolverKind::Eo,
            taus: vec![1.2, 1.4, 1.6],
        };
        // 2 iters x 3 taus = 6（thetas/ks は無視される）。
        let cfgs = sweep.expand();
        assert_eq!(cfgs.len(), 6);
        for c in &cfgs {
            assert_eq!(c.name, c.id());
            assert!(matches!(c.solver, SolverSpec::Eo { .. }));
            assert_eq!(c.theta, None);
            assert_eq!(c.smoothing, SmoothingSpec::None);
        }
        assert_eq!(cfgs[0].solver, SolverSpec::Eo { tau: 1.2 });

        // taus が空なら DEFAULT_TAU の 1 通り。
        let default_sweep = ConfigSweep {
            thetas: vec![],
            log10_iterations: vec![3],
            smoothing_kind: SmoothingKind::None,
            ks: vec![],
            weights: vec![],
            solver_kind: SolverKind::Eo,
            taus: vec![],
        };
        let d_cfgs = default_sweep.expand();
        assert_eq!(d_cfgs.len(), 1);
        assert_eq!(d_cfgs[0].solver, SolverSpec::Eo { tau: DEFAULT_TAU });
    }

    #[test]
    fn test_config_sweep_expand_eoflip() {
        let sweep = ConfigSweep {
            thetas: vec![],
            log10_iterations: vec![4, 5],
            smoothing_kind: SmoothingKind::None,
            ks: vec![],
            weights: vec![],
            solver_kind: SolverKind::EoFlip,
            taus: vec![1.3, 1.5],
        };
        // 2 iters x 2 taus = 4
        let cfgs = sweep.expand();
        assert_eq!(cfgs.len(), 4);
        for c in &cfgs {
            assert_eq!(c.name, c.id());
            assert!(matches!(c.solver, SolverSpec::EoFlip { .. }));
        }
        assert_eq!(cfgs[0].solver, SolverSpec::EoFlip { tau: 1.3 });
        assert_eq!(cfgs[0].id(), "eoflip_iter4_tau1p3");
    }
}
