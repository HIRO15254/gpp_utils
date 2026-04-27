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
}
