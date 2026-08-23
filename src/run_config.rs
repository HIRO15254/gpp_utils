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

/// EoFlip の適応度 q に使う係数 α_eo の既定値（`graph_partition::ALPHA` と同値だが独立に管理する）。
pub const DEFAULT_EO_FLIP_ALPHA: f64 = 0.05;

/// EoFlip の適応度 q に使う diff の指数 p の既定値（従来の二乗）。
pub const DEFAULT_EO_FLIP_DIFF_EXP: f64 = 2.0;

fn default_eo_flip_alpha() -> f64 {
    DEFAULT_EO_FLIP_ALPHA
}

fn default_eo_flip_diff_exp() -> f64 {
    DEFAULT_EO_FLIP_DIFF_EXP
}

/// `EoFlipMulAlpha` の係数 α の既定値。α=1 は λ1 が多数派/少数派とも 1（バイアスなし）
/// になる中立値。
pub const DEFAULT_EO_FLIP_MUL_ALPHA: f64 = 1.0;

/// `EoFlipAddBeta` の係数 β の既定値。
pub const DEFAULT_EO_FLIP_ADD_BETA: f64 = 1.0;

fn default_eo_flip_mul_alpha() -> f64 {
    DEFAULT_EO_FLIP_MUL_ALPHA
}

fn default_eo_flip_add_beta() -> f64 {
    DEFAULT_EO_FLIP_ADD_BETA
}

/// ハイパーパラメータ値（α_eo / p）を id 文字列用に整形する（`fmt_tau` と同流儀、`.`→`p`）。
fn fmt_hyper(x: f64) -> String {
    if x.fract().abs() < 1e-9 {
        format!("{}", x as i64)
    } else {
        format!("{}", x).replace('.', "p")
    }
}

/// ソルバー指定。`Sa` は固定温度メトロポリス（既存）、`Eo` は τ-EO（厳密バランスのスワップ）。
///
/// `RunConfig` に `#[serde(default)]` で埋め込まれるため、`solver` フィールドを持たない
/// 既存の JSON（結果・batch・回帰 baseline）は `Sa` として読み込まれる（後方互換）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SolverSpec {
    /// 固定温度シミュレーテッドアニーリング（フリップ近傍、温度は `RunConfig::theta`）。
    Sa,
    /// 固定温度 SA をスワップ近傍（厳密バランス）で実行する版。温度は `RunConfig::theta`。
    /// Eo（スワップ版）と同一の近傍・厳密バランス・ベイスン(=m_best)を共有し、
    /// 受理がメトロポリス（無条件ではない）である点だけが異なる。`smoothing` は無視。
    SaSwap,
    /// τ-Extremal Optimization（厳密バランスのスワップ版）。`tau` はべき乗則指数（既定 [`DEFAULT_TAU`]）。
    Eo { tau: f64 },
    /// τ-Extremal Optimization（フリップ近傍版）。SA と同一の近傍・目的関数・ベイスン算出を共有し、
    /// バランスはペナルティ項で扱う。適応度は g/deg にバランスペナルティを「悪い辺/良い辺」として
    /// 織り込んだもの。`tau` はべき乗則指数（既定 [`DEFAULT_TAU`]）。
    ///
    /// `alpha_eo`（既定 [`DEFAULT_EO_FLIP_ALPHA`]）と `diff_exp`（p、既定 [`DEFAULT_EO_FLIP_DIFF_EXP`]）は
    /// 適応度 q = `alpha_eo·(|diff|^p − |diff_after|^p)` の係数と指数。**目的関数 `score` には影響しない**
    /// （手選択の内部勾配だけを変える）。既定値では従来の `ALPHA·(diff² − diff_after²)` を byte 完全再現する。
    /// 旧 JSON（`alpha_eo`/`diff_exp` フィールドを持たない）は per-field の serde default で既定値が埋まる。
    EoFlip {
        tau: f64,
        #[serde(default = "default_eo_flip_alpha")]
        alpha_eo: f64,
        #[serde(default = "default_eo_flip_diff_exp")]
        diff_exp: f64,
    },
    /// τ-Extremal Optimization（フリップ近傍・乗算 α 版）。適応度 `λ = λ0 · λ1`。
    ///
    /// `λ0` はスワップ版 EO と同じ次数正規化適応度 `g/deg`（[`SolverSpec::Eo`] の λ 計算と
    /// 同一で、バランスペナルティは含まない）。`λ1` は対象頂点が現在多数派集合
    /// （`|自集合| > |反対集合|`）に属すなら `alpha`、少数派なら `1.0`。両集合が同数
    /// （`cur_t == cur_f`）のときはどちらの集合も多数派扱いしない（`λ1 = 1.0` 固定）。
    /// `alpha` の既定値は [`DEFAULT_EO_FLIP_MUL_ALPHA`]（=1、多数派バイアスなしの中立値）。
    EoFlipMulAlpha {
        tau: f64,
        #[serde(default = "default_eo_flip_mul_alpha")]
        alpha: f64,
    },
    /// τ-Extremal Optimization（フリップ近傍・加算 β 版）。適応度 `λ = β·λ0 + λ1`。
    ///
    /// `λ0` は [`SolverSpec::EoFlipMulAlpha`] と同じスワップ版 EO 由来の `g/deg`。`λ1` は
    /// 多数派なら `0.0`、少数派なら `1.0`（均衡時はどちらも多数派扱いしないため `λ1 = 1.0`）。
    /// `beta` の既定値は [`DEFAULT_EO_FLIP_ADD_BETA`]。
    EoFlipAddBeta {
        tau: f64,
        #[serde(default = "default_eo_flip_add_beta")]
        beta: f64,
    },
    /// τ-Extremal Optimization（フリップ近傍・乗算 γ 版）。適応度 `λ = λ0 · λ1`。
    ///
    /// `λ0` は [`SolverSpec::EoFlipMulAlpha`] と同じ。`λ1` は多数派なら
    /// `γ = (少数派集合の頂点数) / (N/2)`（毎ステップ動的に算出、均衡時は `γ = 1.0`）、
    /// 少数派なら `1.0`。追加のハイパーパラメータはない。
    EoFlipMulGamma { tau: f64 },
}

impl Default for SolverSpec {
    fn default() -> Self {
        SolverSpec::Sa
    }
}

impl SolverSpec {
    /// EO 系ソルバーのべき乗則指数 τ（EO 系でなければ `None`）。
    pub fn tau(&self) -> Option<f64> {
        match *self {
            SolverSpec::Eo { tau }
            | SolverSpec::EoFlip { tau, .. }
            | SolverSpec::EoFlipMulAlpha { tau, .. }
            | SolverSpec::EoFlipAddBeta { tau, .. }
            | SolverSpec::EoFlipMulGamma { tau } => Some(tau),
            SolverSpec::Sa | SolverSpec::SaSwap => None,
        }
    }
}

/// EoFlip 系ソルバーの適応度計算方式（τ を含まない）。
///
/// `run_eo_flip` 内部のディスパッチに使う。τ はランク抽選側のパラメータで
/// 順位付けには関与しないため、ここには含めない。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EoFlipFitnessSpec {
    /// 従来（ペナルティ織り込み）方式。`alpha_eo`・`diff_exp` は [`SolverSpec::EoFlip`] と同じ。
    Legacy { alpha_eo: f64, diff_exp: f64 },
    /// 乗算 α 版（[`SolverSpec::EoFlipMulAlpha`]）。
    MulAlpha { alpha: f64 },
    /// 加算 β 版（[`SolverSpec::EoFlipAddBeta`]）。
    AddBeta { beta: f64 },
    /// 乗算 γ 版（[`SolverSpec::EoFlipMulGamma`]、γ は毎ステップ動的算出）。
    MulGamma,
}

impl EoFlipFitnessSpec {
    /// 解析用の正準ラベル。例: `legacy_a0p064_p2` / `mulalpha_a0p1` / `addbeta_b0` / `mulgamma`。
    ///
    /// `RunConfig::id()` とは**異なる**（id は Legacy の既定値 α_eo/p を省略する）。
    /// 解析側は必ず保存された config から本ラベルを導出し、ディレクトリ名の正規表現には
    /// 依存しないこと。
    pub fn label(&self) -> String {
        match *self {
            Self::Legacy { alpha_eo, diff_exp } => {
                format!("legacy_a{}_p{}", fmt_hyper(alpha_eo), fmt_hyper(diff_exp))
            }
            Self::MulAlpha { alpha } => format!("mulalpha_a{}", fmt_hyper(alpha)),
            Self::AddBeta { beta } => format!("addbeta_b{}", fmt_hyper(beta)),
            Self::MulGamma => "mulgamma".to_string(),
        }
    }

    /// `SolverSpec` から適応度部分を抜き出す（EoFlip 系でなければ `None`）。
    pub fn from_solver(solver: &SolverSpec) -> Option<Self> {
        match *solver {
            SolverSpec::EoFlip { alpha_eo, diff_exp, .. } => {
                Some(Self::Legacy { alpha_eo, diff_exp })
            }
            SolverSpec::EoFlipMulAlpha { alpha, .. } => Some(Self::MulAlpha { alpha }),
            SolverSpec::EoFlipAddBeta { beta, .. } => Some(Self::AddBeta { beta }),
            SolverSpec::EoFlipMulGamma { .. } => Some(Self::MulGamma),
            SolverSpec::Sa | SolverSpec::SaSwap | SolverSpec::Eo { .. } => None,
        }
    }
}

/// ソルバーの種別（パラメータを持たない）。sweep で「種別 × 複数 τ」を表すのに使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverKind {
    Sa,
    SaSwap,
    Eo,
    EoFlip,
    EoFlipMulAlpha,
    EoFlipAddBeta,
    EoFlipMulGamma,
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
                format!(
                    "{}_iter{}_{}",
                    fmt_theta(self.theta),
                    self.log10_iterations,
                    self.smoothing.label()
                )
            }
            SolverSpec::SaSwap => {
                format!("saswap_{}_iter{}", fmt_theta(self.theta), self.log10_iterations)
            }
            SolverSpec::Eo { tau } => {
                format!("eo_iter{}_tau{}", self.log10_iterations, fmt_tau(tau))
            }
            SolverSpec::EoFlip { tau, alpha_eo, diff_exp } => {
                // 既定値のときは従来 id を厳密維持し、非既定のときだけ接尾辞（tau→a→p の順）を付ける。
                let mut s =
                    format!("eoflip_iter{}_tau{}", self.log10_iterations, fmt_tau(tau));
                if (alpha_eo - DEFAULT_EO_FLIP_ALPHA).abs() >= 1e-12 {
                    s.push_str(&format!("_a{}", fmt_hyper(alpha_eo)));
                }
                if (diff_exp - DEFAULT_EO_FLIP_DIFF_EXP).abs() >= 1e-12 {
                    s.push_str(&format!("_p{}", fmt_hyper(diff_exp)));
                }
                s
            }
            SolverSpec::EoFlipMulAlpha { tau, alpha } => {
                format!(
                    "eoflipmulalpha_iter{}_tau{}_a{}",
                    self.log10_iterations,
                    fmt_tau(tau),
                    fmt_hyper(alpha)
                )
            }
            SolverSpec::EoFlipAddBeta { tau, beta } => {
                format!(
                    "eoflipaddbeta_iter{}_tau{}_b{}",
                    self.log10_iterations,
                    fmt_tau(tau),
                    fmt_hyper(beta)
                )
            }
            SolverSpec::EoFlipMulGamma { tau } => {
                format!("eoflipmulgamma_iter{}_tau{}", self.log10_iterations, fmt_tau(tau))
            }
        }
    }
}

/// 温度 Θ を id 文字列用に整形する。`None`→`T0`、整数→`th{+/-N}`、小数→`th{+/-N.NN}`（`.`→`p`）。
fn fmt_theta(theta: Option<f64>) -> String {
    match theta {
        None => "T0".to_string(),
        Some(t) => {
            if (t.fract()).abs() < 1e-9 {
                format!("th{:+}", t as i64)
            } else {
                format!("th{:+.2}", t).replace('.', "p")
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
    /// EoFlip の適応度係数 α_eo 候補（`solver_kind` が `EoFlip` のとき直積軸に使う。
    /// 空なら [`DEFAULT_EO_FLIP_ALPHA`] 1 通り。`EoFlip` 以外では無視）。
    #[serde(default)]
    pub alpha_eos: Vec<f64>,
    /// EoFlip の適応度指数 p（diff_exp）候補（`solver_kind` が `EoFlip` のとき直積軸に使う。
    /// 空なら [`DEFAULT_EO_FLIP_DIFF_EXP`] 1 通り。`EoFlip` 以外では無視）。
    #[serde(default)]
    pub diff_exps: Vec<f64>,
    /// EoFlipMulAlpha の係数 α 候補（`solver_kind` が `EoFlipMulAlpha` のとき直積軸に使う。
    /// 空なら [`DEFAULT_EO_FLIP_MUL_ALPHA`] 1 通り。それ以外では無視）。
    #[serde(default)]
    pub mul_alphas: Vec<f64>,
    /// EoFlipAddBeta の係数 β 候補（`solver_kind` が `EoFlipAddBeta` のとき直積軸に使う。
    /// 空なら [`DEFAULT_EO_FLIP_ADD_BETA`] 1 通り。それ以外では無視）。
    #[serde(default)]
    pub add_betas: Vec<f64>,
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
            SolverKind::SaSwap => self.expand_sa_swap(),
            SolverKind::Eo => self.expand_eo(false),
            SolverKind::EoFlip => self.expand_eo(true),
            SolverKind::EoFlipMulAlpha => self.expand_eo_flip_mul_alpha(),
            SolverKind::EoFlipAddBeta => self.expand_eo_flip_add_beta(),
            SolverKind::EoFlipMulGamma => self.expand_eo_flip_mul_gamma(),
        }
    }

    /// SaSwap: `thetas × log10_iterations`（smoothing なし）を展開する。
    fn expand_sa_swap(&self) -> Vec<RunConfig> {
        let mut out = Vec::new();
        for &theta in &self.thetas {
            for &log10_iterations in &self.log10_iterations {
                let mut cfg = RunConfig {
                    name: String::new(),
                    theta,
                    log10_iterations,
                    smoothing: SmoothingSpec::None,
                    solver: SolverSpec::SaSwap,
                };
                cfg.name = cfg.id();
                out.push(cfg);
            }
        }
        out
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
        // α_eo / p は EoFlip でのみ直積軸になる。非 flip（Eo）や空 Vec は既定 1 通り
        // （空 taus → [DEFAULT_TAU] と同じ規約）。→ Eo の展開数・既定 EoFlip の展開数は不変。
        let alpha_eos: Vec<f64> = if !flip || self.alpha_eos.is_empty() {
            vec![DEFAULT_EO_FLIP_ALPHA]
        } else {
            self.alpha_eos.clone()
        };
        let diff_exps: Vec<f64> = if !flip || self.diff_exps.is_empty() {
            vec![DEFAULT_EO_FLIP_DIFF_EXP]
        } else {
            self.diff_exps.clone()
        };
        let mut out = Vec::new();
        for &log10_iterations in &self.log10_iterations {
            for &tau in &taus {
                for &alpha_eo in &alpha_eos {
                    for &diff_exp in &diff_exps {
                        let solver = if flip {
                            SolverSpec::EoFlip { tau, alpha_eo, diff_exp }
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
            }
        }
        out
    }

    /// EoFlipMulAlpha: `log10_iterations × taus × mul_alphas` を展開する。
    fn expand_eo_flip_mul_alpha(&self) -> Vec<RunConfig> {
        let taus: Vec<f64> = if self.taus.is_empty() { vec![DEFAULT_TAU] } else { self.taus.clone() };
        let alphas: Vec<f64> = if self.mul_alphas.is_empty() {
            vec![DEFAULT_EO_FLIP_MUL_ALPHA]
        } else {
            self.mul_alphas.clone()
        };
        let mut out = Vec::new();
        for &log10_iterations in &self.log10_iterations {
            for &tau in &taus {
                for &alpha in &alphas {
                    let mut cfg = RunConfig {
                        name: String::new(),
                        theta: None,
                        log10_iterations,
                        smoothing: SmoothingSpec::None,
                        solver: SolverSpec::EoFlipMulAlpha { tau, alpha },
                    };
                    cfg.name = cfg.id();
                    out.push(cfg);
                }
            }
        }
        out
    }

    /// EoFlipAddBeta: `log10_iterations × taus × add_betas` を展開する。
    fn expand_eo_flip_add_beta(&self) -> Vec<RunConfig> {
        let taus: Vec<f64> = if self.taus.is_empty() { vec![DEFAULT_TAU] } else { self.taus.clone() };
        let betas: Vec<f64> = if self.add_betas.is_empty() {
            vec![DEFAULT_EO_FLIP_ADD_BETA]
        } else {
            self.add_betas.clone()
        };
        let mut out = Vec::new();
        for &log10_iterations in &self.log10_iterations {
            for &tau in &taus {
                for &beta in &betas {
                    let mut cfg = RunConfig {
                        name: String::new(),
                        theta: None,
                        log10_iterations,
                        smoothing: SmoothingSpec::None,
                        solver: SolverSpec::EoFlipAddBeta { tau, beta },
                    };
                    cfg.name = cfg.id();
                    out.push(cfg);
                }
            }
        }
        out
    }

    /// EoFlipMulGamma: `log10_iterations × taus` を展開する（γ は動的算出のためスイープ軸なし）。
    fn expand_eo_flip_mul_gamma(&self) -> Vec<RunConfig> {
        let taus: Vec<f64> = if self.taus.is_empty() { vec![DEFAULT_TAU] } else { self.taus.clone() };
        let mut out = Vec::new();
        for &log10_iterations in &self.log10_iterations {
            for &tau in &taus {
                let mut cfg = RunConfig {
                    name: String::new(),
                    theta: None,
                    log10_iterations,
                    smoothing: SmoothingSpec::None,
                    solver: SolverSpec::EoFlipMulGamma { tau },
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
            solver: SolverSpec::EoFlip {
                tau: 1.4,
                alpha_eo: DEFAULT_EO_FLIP_ALPHA,
                diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
            },
        };
        assert_eq!(c3.id(), "eoflip_iter5_tau1p4");
        // SaSwap は theta を使い smoothing を含めない。
        let c4 = RunConfig {
            name: "a".into(),
            theta: Some(0.0),
            log10_iterations: 4,
            smoothing: SmoothingSpec::KAverage(8),
            solver: SolverSpec::SaSwap,
        };
        assert_eq!(c4.id(), "saswap_th+0_iter4");
        let c5 = RunConfig {
            name: "a".into(),
            theta: None,
            log10_iterations: 5,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::SaSwap,
        };
        assert_eq!(c5.id(), "saswap_T0_iter5");
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

        // 旧 EoFlip JSON（alpha_eo/diff_exp なし）は per-field serde default で既定値が埋まり、
        // id も従来どおり接尾辞なし（既存 data/results と再現性を保つ）。
        let json_flip_old =
            r#"{"name":"x","theta":null,"log10_iterations":5,"smoothing":"None","solver":{"EoFlip":{"tau":1.4}}}"#;
        let cfg_flip_old: RunConfig =
            serde_json::from_str(json_flip_old).expect("deserialize old eoflip");
        assert_eq!(
            cfg_flip_old.solver,
            SolverSpec::EoFlip {
                tau: 1.4,
                alpha_eo: DEFAULT_EO_FLIP_ALPHA,
                diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
            }
        );
        assert_eq!(cfg_flip_old.id(), "eoflip_iter5_tau1p4");

        // 明示的な alpha_eo/diff_exp を持つ EoFlip JSON は round-trip できる。
        let json_flip_full = r#"{"name":"x","theta":null,"log10_iterations":5,"smoothing":"None","solver":{"EoFlip":{"tau":1.4,"alpha_eo":0.1,"diff_exp":0.5}}}"#;
        let cfg_flip_full: RunConfig =
            serde_json::from_str(json_flip_full).expect("deserialize full eoflip");
        assert_eq!(
            cfg_flip_full.solver,
            SolverSpec::EoFlip { tau: 1.4, alpha_eo: 0.1, diff_exp: 0.5 }
        );
        assert_eq!(cfg_flip_full.id(), "eoflip_iter5_tau1p4_a0p1_p0p5");
    }

    #[test]
    fn test_id_format_eoflip_hyper() {
        let mk = |alpha_eo: f64, diff_exp: f64| RunConfig {
            name: "a".into(),
            theta: None,
            log10_iterations: 5,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::EoFlip { tau: 1.4, alpha_eo, diff_exp },
        };
        // 既定 → 接尾辞なし（従来 id を厳密維持）。
        assert_eq!(
            mk(DEFAULT_EO_FLIP_ALPHA, DEFAULT_EO_FLIP_DIFF_EXP).id(),
            "eoflip_iter5_tau1p4"
        );
        // 両方非既定 → tau→a→p の順で接尾辞。
        assert_eq!(mk(0.1, 0.5).id(), "eoflip_iter5_tau1p4_a0p1_p0p5");
        // p だけ非既定 → _p のみ。
        assert_eq!(mk(DEFAULT_EO_FLIP_ALPHA, 0.5).id(), "eoflip_iter5_tau1p4_p0p5");
        // α だけ非既定 → _a のみ。
        assert_eq!(mk(0.2, DEFAULT_EO_FLIP_DIFF_EXP).id(), "eoflip_iter5_tau1p4_a0p2");
    }

    #[test]
    fn test_config_sweep_expand_eoflip_hyper() {
        // EoFlip の 3 次元スイープ: taus × alpha_eos × diff_exps（× iters=1）。
        let sweep = ConfigSweep {
            thetas: vec![],
            log10_iterations: vec![4],
            smoothing_kind: SmoothingKind::None,
            ks: vec![],
            weights: vec![],
            solver_kind: SolverKind::EoFlip,
            taus: vec![1.3, 1.5],
            alpha_eos: vec![0.05, 0.1],
            diff_exps: vec![2.0, 0.5],
            mul_alphas: vec![],
            add_betas: vec![],
        };
        let cfgs = sweep.expand();
        // 2 taus × 2 alpha_eos × 2 diff_exps = 8。
        assert_eq!(cfgs.len(), 8);
        // id はすべて相異（name = id）。
        let ids: std::collections::HashSet<_> = cfgs.iter().map(|c| c.name.clone()).collect();
        assert_eq!(ids.len(), 8, "8 config は一意な id を持つ");
        for c in &cfgs {
            assert_eq!(c.name, c.id());
            assert!(matches!(c.solver, SolverSpec::EoFlip { .. }));
        }

        // 空 α/p Vec なら taus のみの展開数（既定 1 通り）。
        let sweep_empty = ConfigSweep {
            thetas: vec![],
            log10_iterations: vec![4],
            smoothing_kind: SmoothingKind::None,
            ks: vec![],
            weights: vec![],
            solver_kind: SolverKind::EoFlip,
            taus: vec![1.3, 1.5],
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
        };
        let cfgs_empty = sweep_empty.expand();
        assert_eq!(cfgs_empty.len(), 2);
        assert_eq!(
            cfgs_empty[0].solver,
            SolverSpec::EoFlip {
                tau: 1.3,
                alpha_eo: DEFAULT_EO_FLIP_ALPHA,
                diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
            }
        );
        assert_eq!(cfgs_empty[0].id(), "eoflip_iter4_tau1p3");

        // 非 flip（Eo）は α/p 軸を無視する（展開数は taus のみ）。
        let sweep_eo = ConfigSweep {
            thetas: vec![],
            log10_iterations: vec![4],
            smoothing_kind: SmoothingKind::None,
            ks: vec![],
            weights: vec![],
            solver_kind: SolverKind::Eo,
            taus: vec![1.3, 1.5],
            alpha_eos: vec![0.05, 0.1, 0.2],
            diff_exps: vec![2.0, 0.5],
            mul_alphas: vec![],
            add_betas: vec![],
        };
        assert_eq!(sweep_eo.expand().len(), 2, "Eo は α/p を無視");
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
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
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
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
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
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
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
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
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
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
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
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
        };
        // 2 iters x 2 taus = 4
        let cfgs = sweep.expand();
        assert_eq!(cfgs.len(), 4);
        for c in &cfgs {
            assert_eq!(c.name, c.id());
            assert!(matches!(c.solver, SolverSpec::EoFlip { .. }));
        }
        assert_eq!(
            cfgs[0].solver,
            SolverSpec::EoFlip {
                tau: 1.3,
                alpha_eo: DEFAULT_EO_FLIP_ALPHA,
                diff_exp: DEFAULT_EO_FLIP_DIFF_EXP,
            }
        );
        assert_eq!(cfgs[0].id(), "eoflip_iter4_tau1p3");
    }

    #[test]
    fn test_config_sweep_expand_sa_swap() {
        let sweep = ConfigSweep {
            thetas: vec![Some(-1.0), Some(0.0), None],
            log10_iterations: vec![4, 5],
            smoothing_kind: SmoothingKind::KAverage,
            ks: vec![4, 8],
            weights: vec![],
            solver_kind: SolverKind::SaSwap,
            taus: vec![],
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
        };
        // 3 thetas x 2 iters = 6（smoothing/ks は無視）。
        let cfgs = sweep.expand();
        assert_eq!(cfgs.len(), 6);
        for c in &cfgs {
            assert_eq!(c.name, c.id());
            assert_eq!(c.solver, SolverSpec::SaSwap);
            assert_eq!(c.smoothing, SmoothingSpec::None);
        }
        assert_eq!(cfgs[0].id(), "saswap_th-1_iter4");
    }

    #[test]
    fn test_id_format_eoflip_mul_alpha() {
        let c = RunConfig {
            name: "a".into(),
            theta: None,
            log10_iterations: 5,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::EoFlipMulAlpha { tau: 1.4, alpha: 0.3 },
        };
        assert_eq!(c.id(), "eoflipmulalpha_iter5_tau1p4_a0p3");
    }

    #[test]
    fn test_id_format_eoflip_add_beta() {
        let c = RunConfig {
            name: "a".into(),
            theta: None,
            log10_iterations: 5,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::EoFlipAddBeta { tau: 1.4, beta: 2.0 },
        };
        assert_eq!(c.id(), "eoflipaddbeta_iter5_tau1p4_b2");
    }

    #[test]
    fn test_id_format_eoflip_mul_gamma() {
        let c = RunConfig {
            name: "a".into(),
            theta: None,
            log10_iterations: 5,
            smoothing: SmoothingSpec::None,
            solver: SolverSpec::EoFlipMulGamma { tau: 1.4 },
        };
        assert_eq!(c.id(), "eoflipmulgamma_iter5_tau1p4");
    }

    #[test]
    fn test_config_sweep_expand_eoflip_mul_alpha() {
        let sweep = ConfigSweep {
            thetas: vec![],
            log10_iterations: vec![4, 5],
            smoothing_kind: SmoothingKind::None,
            ks: vec![],
            weights: vec![],
            solver_kind: SolverKind::EoFlipMulAlpha,
            taus: vec![1.3, 1.5],
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![0.2, 0.5],
            add_betas: vec![],
        };
        // 2 iters x 2 taus x 2 alphas = 8
        let cfgs = sweep.expand();
        assert_eq!(cfgs.len(), 8);
        for c in &cfgs {
            assert_eq!(c.name, c.id());
            assert!(matches!(c.solver, SolverSpec::EoFlipMulAlpha { .. }));
        }
        assert_eq!(cfgs[0].solver, SolverSpec::EoFlipMulAlpha { tau: 1.3, alpha: 0.2 });

        // 空 mul_alphas なら既定 1 通り。
        let sweep_empty = ConfigSweep {
            thetas: vec![],
            log10_iterations: vec![4],
            smoothing_kind: SmoothingKind::None,
            ks: vec![],
            weights: vec![],
            solver_kind: SolverKind::EoFlipMulAlpha,
            taus: vec![1.4],
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
        };
        let cfgs_empty = sweep_empty.expand();
        assert_eq!(cfgs_empty.len(), 1);
        assert_eq!(
            cfgs_empty[0].solver,
            SolverSpec::EoFlipMulAlpha { tau: 1.4, alpha: DEFAULT_EO_FLIP_MUL_ALPHA }
        );
    }

    #[test]
    fn test_config_sweep_expand_eoflip_add_beta() {
        let sweep = ConfigSweep {
            thetas: vec![],
            log10_iterations: vec![4],
            smoothing_kind: SmoothingKind::None,
            ks: vec![],
            weights: vec![],
            solver_kind: SolverKind::EoFlipAddBeta,
            taus: vec![1.3, 1.5],
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![0.5, 1.5],
        };
        // 1 iter x 2 taus x 2 betas = 4
        let cfgs = sweep.expand();
        assert_eq!(cfgs.len(), 4);
        for c in &cfgs {
            assert_eq!(c.name, c.id());
            assert!(matches!(c.solver, SolverSpec::EoFlipAddBeta { .. }));
        }
        assert_eq!(cfgs[0].solver, SolverSpec::EoFlipAddBeta { tau: 1.3, beta: 0.5 });

        // 空 add_betas なら既定 1 通り。
        let sweep_empty = ConfigSweep {
            thetas: vec![],
            log10_iterations: vec![4],
            smoothing_kind: SmoothingKind::None,
            ks: vec![],
            weights: vec![],
            solver_kind: SolverKind::EoFlipAddBeta,
            taus: vec![1.4],
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
        };
        let cfgs_empty = sweep_empty.expand();
        assert_eq!(cfgs_empty.len(), 1);
        assert_eq!(
            cfgs_empty[0].solver,
            SolverSpec::EoFlipAddBeta { tau: 1.4, beta: DEFAULT_EO_FLIP_ADD_BETA }
        );
    }

    #[test]
    fn test_eo_id_has_no_tie_suffix() {
        // tie 規則が 1 本化されたので id にモード接尾辞は付かない。
        let mut c = RunConfig::new("a");
        c.theta = None;
        c.log10_iterations = 6;
        c.solver = SolverSpec::EoFlipMulGamma { tau: 1.4 };
        assert_eq!(c.id(), "eoflipmulgamma_iter6_tau1p4");

        let sa = RunConfig::new("a");
        assert_eq!(sa.id(), "th+0_iter4_none");
        let mut eo = RunConfig::new("a");
        eo.solver = SolverSpec::Eo { tau: 1.4 };
        assert_eq!(eo.id(), "eo_iter4_tau1p4");

        // 旧 JSON に残る `eo_tie_break` は未知フィールドとして無視される（後方互換）。
        let json = r#"{"name":"x","theta":null,"log10_iterations":5,"smoothing":"None","solver":{"EoFlip":{"tau":1.4}},"eo_tie_break":"CompetitionRandom"}"#;
        let cfg: RunConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(cfg.id(), "eoflip_iter5_tau1p4");
    }

    #[test]
    fn test_config_sweep_expand_eoflip_mul_gamma() {
        let sweep = ConfigSweep {
            thetas: vec![],
            log10_iterations: vec![4, 5],
            smoothing_kind: SmoothingKind::None,
            ks: vec![],
            weights: vec![],
            solver_kind: SolverKind::EoFlipMulGamma,
            taus: vec![1.3, 1.5],
            alpha_eos: vec![],
            diff_exps: vec![],
            mul_alphas: vec![],
            add_betas: vec![],
        };
        // 2 iters x 2 taus = 4（γ はスイープ軸を持たない）。
        let cfgs = sweep.expand();
        assert_eq!(cfgs.len(), 4);
        for c in &cfgs {
            assert_eq!(c.name, c.id());
            assert!(matches!(c.solver, SolverSpec::EoFlipMulGamma { .. }));
        }
        assert_eq!(cfgs[0].solver, SolverSpec::EoFlipMulGamma { tau: 1.3 });
        assert_eq!(cfgs[0].id(), "eoflipmulgamma_iter4_tau1p3");
    }
}
