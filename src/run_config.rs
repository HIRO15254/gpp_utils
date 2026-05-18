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
    /// 重み付き平均（K/n × avg + (1-K/n) × current）。
    WeightedAverage(usize),
}

impl SmoothingSpec {
    pub fn label(&self) -> String {
        match self {
            Self::None => "none".into(),
            Self::KAverage(k) => format!("kavg{}", k),
            Self::RandomKAverage(k) => format!("rkavg{}", k),
            Self::WeightedAverage(k) => format!("wavg{}", k),
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
}

impl RunConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            theta: Some(0.0),
            log10_iterations: 4,
            smoothing: SmoothingSpec::None,
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
    pub fn id(&self) -> String {
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
}

/// 平滑化の種別（K 値を持たない）。sweep 指定で「種別 × 複数 K」を表すのに使う。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmoothingKind {
    None,
    KAverage,
    RandomKAverage,
    WeightedAverage,
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

    /// K 値を必要とするか（`None` 以外は必要）。
    pub fn needs_k(self) -> bool {
        !matches!(self, Self::None)
    }

    /// 種別と K から `SmoothingSpec` を作る。`None` 種別では K を無視する。
    pub fn with_k(self, k: usize) -> SmoothingSpec {
        match self {
            Self::None => SmoothingSpec::None,
            Self::KAverage => SmoothingSpec::KAverage(k),
            Self::RandomKAverage => SmoothingSpec::RandomKAverage(k),
            Self::WeightedAverage => SmoothingSpec::WeightedAverage(k),
        }
    }
}

/// 温度・反復回数・平滑化 K の各軸を複数指定し、その直積で `RunConfig` 群を生成する指定。
///
/// 平滑化は「単一種別 × 複数 K」の形をとる（`smoothing_kind` が `None` のときは
/// `ks` を無視し、平滑化なしの 1 通りとなる）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSweep {
    /// 温度 Θ の候補。`null` は T = 0（貪欲）。
    pub thetas: Vec<Option<f64>>,
    /// 反復回数指数 N（max_iter = 10^N）の候補。
    pub log10_iterations: Vec<u32>,
    /// 平滑化の種別（生成される全 `RunConfig` で共通）。
    pub smoothing_kind: SmoothingKind,
    /// 平滑化の K 値候補。`smoothing_kind` が `None` のときは無視される。
    #[serde(default)]
    pub ks: Vec<usize>,
}

impl ConfigSweep {
    /// `thetas × log10_iterations × smoothings` の直積を取り `RunConfig` 群を生成する。
    /// 各 `RunConfig` の `name` には一意な `id()` 文字列が入る。
    pub fn expand(&self) -> Vec<RunConfig> {
        let smoothings: Vec<SmoothingSpec> = if self.smoothing_kind.needs_k() {
            self.ks.iter().map(|&k| self.smoothing_kind.with_k(k)).collect()
        } else {
            vec![SmoothingSpec::None]
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
                    };
                    cfg.name = cfg.id();
                    out.push(cfg);
                }
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
        };
        assert_eq!(c.id(), "th+0_iter4_kavg8");
        let c0 = RunConfig {
            name: "a".into(),
            theta: None,
            log10_iterations: 5,
            smoothing: SmoothingSpec::None,
        };
        assert_eq!(c0.id(), "T0_iter5_none");
    }

    #[test]
    fn test_config_sweep_expand() {
        let sweep = ConfigSweep {
            thetas: vec![Some(0.0), None],
            log10_iterations: vec![4, 5],
            smoothing_kind: SmoothingKind::KAverage,
            ks: vec![4, 8],
        };
        // 2 thetas x 2 iters x 2 Ks = 8
        let cfgs = sweep.expand();
        assert_eq!(cfgs.len(), 8);
        // 生成された RunConfig の name は id() と一致する。
        for c in &cfgs {
            assert_eq!(c.name, c.id());
        }

        // None 種別は ks を無視し、平滑化なしの 1 通りに展開される。
        let none_sweep = ConfigSweep {
            thetas: vec![Some(1.0)],
            log10_iterations: vec![3],
            smoothing_kind: SmoothingKind::None,
            ks: vec![4, 8, 16],
        };
        let none_cfgs = none_sweep.expand();
        assert_eq!(none_cfgs.len(), 1);
        assert_eq!(none_cfgs[0].smoothing, SmoothingSpec::None);
    }
}
