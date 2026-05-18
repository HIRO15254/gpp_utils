//! スコア計算方法の差し替え実装。
//!
//! 同じ問題に対して異なるスコア評価方法（平滑化戦略）を提供する。

use std::sync::Mutex;

use rand::Rng;
use rand_mt::Mt19937GenRand64;

use crate::optimization::{Problem, Smoothing};

/// スムージングなし（元のスコアをそのまま使用）。
#[derive(Debug, Clone)]
pub struct NoSmoothing;

impl<S: Clone> Smoothing<S> for NoSmoothing {
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64 {
        problem.score(solution)
    }
}

/// K-近傍平均によるスムージング。
///
/// ランダムに選んだ K 個の近傍のスコアを平均して、
/// 平滑化されたスコアを計算する。
#[derive(Debug, Clone)]
pub struct KAveragingSmoothing {
    /// サンプリングする近傍数。
    pub k: usize,
}

impl KAveragingSmoothing {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl<S: Clone> Smoothing<S> for KAveragingSmoothing {
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64 {
        let n = problem.neighbour_size();
        if n == 0 {
            return problem.score(solution);
        }
        let sample_count = self.k.min(n);
        // 元実装の `neighbours.iter().take(sample_count).map(|n| problem.score(n)).sum()` と
        // 等価。インデックス 0..sample_count を順に評価する。
        let sum: f64 = (0..sample_count)
            .map(|i| problem.score_at_move(solution, i))
            .sum();
        sum / sample_count as f64
    }
}

/// 決定論的な全近傍平均スムージング。
///
/// ランダムサンプリングではなく、すべての近傍のスコアを平均する。
/// 計算コストは高いが、より安定した評価が得られる。
#[derive(Debug, Clone)]
pub struct AllNeighbourAveragingSmoothing;

impl<S: Clone> Smoothing<S> for AllNeighbourAveragingSmoothing {
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64 {
        let n = problem.neighbour_size();
        if n == 0 {
            return problem.score(solution);
        }
        let sum: f64 = (0..n).map(|i| problem.score_at_move(solution, i)).sum();
        sum / n as f64
    }
}

/// ランダムK-近傍平均スムージング（距離2近傍フォールバック付き）。
///
/// 距離1の近傍からランダムにK個サンプリングして平均する。
/// K が距離1の近傍数を超えた場合は、距離1の近傍をすべて使った上で
/// 不足分を距離2の近傍（2ステップ先の解）からランダムに補充する。
///
/// `Smoothing::score()` は呼び出しのたびに内部RNGを進めるため、
/// 同じ解でも異なるスコアを返すことがある（確率的スムージング）。
///
/// # Type bounds
/// `S: PartialEq` が必要（距離2近傍の重複排除に使用）。
pub struct RandomKSmoothing {
    /// サンプリングする近傍数。
    pub k: usize,
    rng: Mutex<Mt19937GenRand64>,
}

impl RandomKSmoothing {
    /// 新しいスムージングを作成する。`seed` で乱数列を固定できる。
    pub fn new(k: usize, seed: u64) -> Self {
        Self { k, rng: Mutex::new(Mt19937GenRand64::new(seed)) }
    }
}

impl std::fmt::Debug for RandomKSmoothing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomKSmoothing").field("k", &self.k).finish()
    }
}

impl<S: Clone + PartialEq + std::hash::Hash + Eq> Smoothing<S> for RandomKSmoothing {
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64 {
        let n = problem.neighbour_size();
        if n == 0 {
            return problem.score(solution);
        }

        let mut rng = self.rng.lock().unwrap();

        let scores: Vec<f64> = if self.k <= n {
            // --- 距離1近傍からK個をランダムサンプリング（非復元） ---
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..self.k {
                let j = rng.gen_range(i..n);
                indices.swap(i, j);
            }
            // 選ばれた K インデックスに対して score_at_move 評価。
            // 元実装の `d1[i].clone()` を materialize してから score するのと等価。
            indices[..self.k]
                .iter()
                .map(|&i| problem.score_at_move(solution, i))
                .collect()
        } else {
            // --- d1 全部 + 不足分をd2から補充 ---
            // d1 (=N個) はそのまま全部スコアリング。
            // d2 は元実装の enumerate 順で構築するが、HashSet で重複排除して O(N³) に抑える。
            let needed = self.k - n;

            // 元実装と同じ enumerate 順で d2 を構築:
            //   for n1 in &d1 { for n2 in neighbour(n1) { ... } }
            // 重複判定は HashSet<S> で吸収（挿入順は維持される）。
            let d1: Vec<S> = problem.neighbour(solution);
            let mut seen: std::collections::HashSet<S> = d1.iter().cloned().collect();
            seen.insert(solution.clone());
            let mut d2: Vec<S> = Vec::new();
            for n1 in &d1 {
                for n2 in problem.neighbour(n1) {
                    if seen.insert(n2.clone()) {
                        d2.push(n2);
                    }
                }
            }

            // d2 から needed 個を Fisher-Yates でランダムサンプリング。
            let take = needed.min(d2.len());
            for i in 0..take {
                let j = rng.gen_range(i..d2.len());
                d2.swap(i, j);
            }

            // d1 全部 + d2 の先頭 take 個を score。
            let mut scores: Vec<f64> = d1.iter().map(|s| problem.score(s)).collect();
            scores.extend(d2[..take].iter().map(|s| problem.score(s)));
            scores
        };

        if scores.is_empty() {
            return problem.score(solution);
        }
        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

/// 重み付き全近傍平均スムージング。
///
/// K 個をランダムに取る代わりに、**すべての近傍**を使って以下の線形ブレンドで
/// スムージングスコアを計算する:
///
/// ```text
/// score = (K/n) × avg(近傍スコア) + (1 - K/n) × current_score
/// ```
///
/// - K = 0 → `NoSmoothing` と等価（元のスコアをそのまま返す）
/// - K = n → `AllNeighbourAveragingSmoothing` と等価（全近傍の平均）
/// - 0 < K < n → 元スコアと全近傍平均の線形補間
///
/// 決定的（乱数不要）で、K が連続パラメータとして機能する。
#[derive(Debug, Clone)]
pub struct WeightedNeighbourSmoothing {
    /// 重みの分子（K）。K ≤ 近傍数 に自動クランプされる。
    pub k: usize,
}

impl WeightedNeighbourSmoothing {
    /// 新しいスムージングを作成する。
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl<S: Clone> Smoothing<S> for WeightedNeighbourSmoothing {
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64 {
        let n = problem.neighbour_size();
        if n == 0 {
            return problem.score(solution);
        }

        let k = self.k.min(n) as f64;
        let weight = k / n as f64; // K / 全近傍数

        // 元実装の `neighbours.iter().map(|nb| problem.score(nb)).sum() / n` と等価。
        let neighbour_avg = (0..n)
            .map(|i| problem.score_at_move(solution, i))
            .sum::<f64>()
            / n as f64;
        let current_score = problem.score(solution);

        weight * neighbour_avg + (1.0 - weight) * current_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_mt::Mt19937GenRand64;

    #[derive(Clone)]
    struct DummyProblem;

    impl Problem<i32> for DummyProblem {
        fn score(&self, solution: &i32) -> f64 {
            (solution * solution) as f64
        }

        fn neighbour(&self, solution: &i32) -> Vec<i32> {
            vec![solution - 1, *solution, solution + 1]
        }

        fn random_solution(&self, _rng: &mut Mt19937GenRand64) -> i32 {
            0
        }

        fn neighbour_size(&self) -> usize {
            3
        }
    }

    #[test]
    fn test_no_smoothing() {
        let problem = DummyProblem;
        let smoothing = NoSmoothing;
        let solution = 5i32;
        assert_eq!(
            smoothing.score(&problem, &solution),
            problem.score(&solution)
        );
    }

    #[test]
    fn test_k_averaging_smoothing() {
        let problem = DummyProblem;
        let smoothing = KAveragingSmoothing::new(2);
        let solution = 5i32;

        // neighbours = [4, 5, 6]
        // scores = [16, 25, 36]
        // average of first 2: (16 + 25) / 2 = 20.5
        let score = smoothing.score(&problem, &solution);
        assert!((score - 20.5).abs() < 1e-10);
    }

    #[test]
    fn test_all_neighbour_averaging() {
        let problem = DummyProblem;
        let smoothing = AllNeighbourAveragingSmoothing;
        let solution = 5i32;

        // neighbours = [4, 5, 6]
        // scores = [16, 25, 36]
        // average: (16 + 25 + 36) / 3 = 77 / 3 ≈ 25.667
        let score = smoothing.score(&problem, &solution);
        assert!((score - (77.0 / 3.0)).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // RandomKSmoothing
    // -------------------------------------------------------------------------

    // i32 に PartialEq は実装済みなので DummyProblem で使用可能
    #[test]
    fn test_random_k_smoothing_k_within_d1() {
        // K=2 の場合: d1=[4,5,6] からランダムに 2 個選んで平均
        // スコア候補: 16, 25, 36
        // どの 2 個でも平均は 20.5 〜 30.5 の範囲になる
        let problem = DummyProblem;
        let smoothing = RandomKSmoothing::new(2, 42);
        let solution = 5i32;

        let score = smoothing.score(&problem, &solution);
        // possible averages: (16+25)/2=20.5, (16+36)/2=26.0, (25+36)/2=30.5
        let valid = [20.5, 26.0, 30.5];
        assert!(valid.iter().any(|&v| (score - v).abs() < 1e-10),
            "score {} is not a valid 2-subset average", score);
    }

    #[test]
    fn test_random_k_smoothing_k_equals_d1() {
        // K=3 = d1 の全個数: すべての近傍を使う → 全近傍平均と同じ
        let problem = DummyProblem;
        let smoothing = RandomKSmoothing::new(3, 42);
        let solution = 5i32;

        let score = smoothing.score(&problem, &solution);
        // (16 + 25 + 36) / 3 = 77/3
        assert!((score - 77.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_k_smoothing_d2_fallback() {
        // K=5 > d1.len()=3 なので d2 から 2 個補充する
        // d1 = [4, 5, 6] (scores: 16, 25, 36)
        // d2（DummyProblem の場合）= [3, 7] (scores: 9, 49)
        // 合計 5 個: sum = 16+25+36+9+49 = 135, avg = 27.0
        let problem = DummyProblem;
        let smoothing = RandomKSmoothing::new(5, 42);
        let solution = 5i32;

        let score = smoothing.score(&problem, &solution);
        assert!((score - 27.0).abs() < 1e-10,
            "expected 27.0, got {}", score);
    }

    #[test]
    fn test_random_k_smoothing_k_exceeds_d1_plus_d2() {
        // K が d1+d2 合計を超えた場合: 取れるだけ取って平均
        let problem = DummyProblem;
        let smoothing = RandomKSmoothing::new(100, 42);
        let solution = 5i32;

        let score = smoothing.score(&problem, &solution);
        // d1=[4,5,6], d2=[3,7] → 全5個の平均 = 135/5 = 27.0
        assert!((score - 27.0).abs() < 1e-10,
            "expected 27.0 (all d1+d2), got {}", score);
    }

    // -------------------------------------------------------------------------
    // WeightedNeighbourSmoothing
    // -------------------------------------------------------------------------

    #[test]
    fn test_weighted_k0_equals_no_smoothing() {
        // K=0: weight=0/3=0 → pure current score
        let problem = DummyProblem;
        let smoothing = WeightedNeighbourSmoothing::new(0);
        let solution = 5i32;

        let score = smoothing.score(&problem, &solution);
        assert!((score - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_kn_equals_all_neighbour_avg() {
        // K=n(=3): weight=3/3=1 → pure neighbour average = 77/3
        let problem = DummyProblem;
        let smoothing = WeightedNeighbourSmoothing::new(3);
        let solution = 5i32;

        let score = smoothing.score(&problem, &solution);
        assert!((score - 77.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_k_clamp_above_n() {
        // K > n の場合は n にクランプされ、全近傍平均と同じになる
        let problem = DummyProblem;
        let smoothing = WeightedNeighbourSmoothing::new(999);
        let solution = 5i32;

        let score = smoothing.score(&problem, &solution);
        assert!((score - 77.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_k1_is_linear_blend() {
        // K=1: weight=1/3
        // score = (1/3) * (77/3) + (2/3) * 25
        //       = 77/9 + 50/3
        //       = 77/9 + 150/9 = 227/9
        let problem = DummyProblem;
        let smoothing = WeightedNeighbourSmoothing::new(1);
        let solution = 5i32;

        let score = smoothing.score(&problem, &solution);
        let expected = 227.0 / 9.0;
        assert!((score - expected).abs() < 1e-10,
            "expected {}, got {}", expected, score);
    }
}
