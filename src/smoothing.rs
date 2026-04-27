//! スコア計算方法の差し替え実装。
//!
//! 同じ問題に対して異なるスコア評価方法（平滑化戦略）を提供する。

use std::sync::Mutex;

use rand::Rng;
use rand_mt::Mt19937GenRand64;

use crate::optimization::{Problem, Smoothing};

/// スムージングなし（元のスコアをそのまま使用）。
///
/// これは optimization.rs の NoSmoothing と同じ実装ですが、
/// 別モジュールで再度定義することで、独立した使用が可能です。
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
        let neighbours = problem.neighbour(solution);
        if neighbours.is_empty() {
            return problem.score(solution);
        }

        let sample_count = self.k.min(neighbours.len());
        let sum: f64 = neighbours
            .iter()
            .take(sample_count)
            .map(|n| problem.score(n))
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
        let neighbours = problem.neighbour(solution);
        if neighbours.is_empty() {
            return problem.score(solution);
        }

        let sum: f64 = neighbours.iter().map(|n| problem.score(n)).sum();
        sum / neighbours.len() as f64
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

impl<S: Clone + PartialEq> Smoothing<S> for RandomKSmoothing {
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64 {
        let d1 = problem.neighbour(solution);
        if d1.is_empty() {
            return problem.score(solution);
        }

        let mut rng = self.rng.lock().unwrap();

        let samples: Vec<S> = if self.k <= d1.len() {
            // --- 距離1近傍からK個をランダムサンプリング（非復元） ---
            let mut indices: Vec<usize> = (0..d1.len()).collect();
            // Partial Fisher-Yates: 先頭 k 要素を確定させる
            for i in 0..self.k {
                let j = rng.gen_range(i..d1.len());
                indices.swap(i, j);
            }
            indices[..self.k].iter().map(|&i| d1[i].clone()).collect()
        } else {
            // --- d1 全部 + 不足分をd2から補充 ---
            let needed = self.k - d1.len();

            // 距離2近傍を収集（元の解・d1と重複するものを除外）
            let mut d2: Vec<S> = Vec::new();
            for n1 in &d1 {
                for n2 in problem.neighbour(n1) {
                    if &n2 != solution
                        && !d1.iter().any(|x| x == &n2)
                        && !d2.iter().any(|x| x == &n2)
                    {
                        d2.push(n2);
                    }
                }
            }

            let mut samples = d1.clone();
            // d2 から needed 個をランダムサンプリング（非復元）
            let take = needed.min(d2.len());
            for i in 0..take {
                let j = rng.gen_range(i..d2.len());
                d2.swap(i, j);
            }
            samples.extend_from_slice(&d2[..take]);
            samples
        };

        if samples.is_empty() {
            return problem.score(solution);
        }
        samples.iter().map(|s| problem.score(s)).sum::<f64>() / samples.len() as f64
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
        let neighbours = problem.neighbour(solution);
        if neighbours.is_empty() {
            return problem.score(solution);
        }

        let n = neighbours.len();
        let k = self.k.min(n) as f64;
        let weight = k / n as f64; // K / 全近傍数

        let neighbour_avg = neighbours.iter().map(|nb| problem.score(nb)).sum::<f64>() / n as f64;
        let current_score = problem.score(solution);

        weight * neighbour_avg + (1.0 - weight) * current_score
    }
}

/// 複数のスムージングレベルを順序付きで提供する。
///
/// space_id に応じて異なる K 値を使用した K-近傍平均を提供し、
/// 滑らかな空間から元の空間への段階的な移行を実現する。
#[derive(Debug, Clone)]
pub struct MultiLevelSmoothing {
    /// スムージング空間の数。
    space_count: usize,
    /// 最も平滑化された空間での最大 K 値。
    max_k: usize,
}

impl MultiLevelSmoothing {
    /// 新しい multi-level smoothing を作成する。
    ///
    /// # Arguments
    /// - `space_count`: スムージング空間の数（2 以上推奨）
    /// - `max_k`: 空間 0 でのサンプリング数（大きいほど平滑）
    pub fn new(space_count: usize, max_k: usize) -> Self {
        Self { space_count, max_k }
    }

    /// 空間インデックスに対応する K 値を計算する。
    fn k_for_space(&self, space_id: usize) -> usize {
        if self.space_count <= 1 || space_id >= self.space_count - 1 {
            return 1;
        }
        let t = space_id as f64 / (self.space_count - 1) as f64;
        let k = self.max_k as f64 * (1.0 - t) + 1.0 * t;
        k.round().max(1.0) as usize
    }

    /// 指定した空間でのスムージングを実行する。
    pub fn score_in_space<S: Clone>(
        &self,
        space_id: usize,
        problem: &dyn Problem<S>,
        solution: &S,
    ) -> f64 {
        let k = self.k_for_space(space_id);
        if k <= 1 {
            return problem.score(solution);
        }

        let neighbours = problem.neighbour(solution);
        if neighbours.is_empty() {
            return problem.score(solution);
        }

        let sample_count = k.min(neighbours.len());
        let sum: f64 = neighbours
            .iter()
            .take(sample_count)
            .map(|n| problem.score(n))
            .sum();
        sum / sample_count as f64
    }

    /// スムージング空間の数を返す。
    pub fn space_count(&self) -> usize {
        self.space_count
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

    #[test]
    fn test_multi_level_smoothing_k_values() {
        let multi = MultiLevelSmoothing::new(5, 50);
        // space_id=0: k = 50 * (1 - 0) + 1 * 0 = 50
        assert_eq!(multi.k_for_space(0), 50);
        // space_id=1: t = 1/4 = 0.25; k = 50 * 0.75 + 1 * 0.25 = 37.75 → 38
        assert_eq!(multi.k_for_space(1), 38);
        // space_id=4: space_id >= space_count - 1, so k = 1
        assert_eq!(multi.k_for_space(4), 1);
    }

    #[test]
    fn test_multi_level_smoothing_space_count() {
        let multi = MultiLevelSmoothing::new(10, 100);
        assert_eq!(multi.space_count(), 10);
    }

    #[test]
    fn test_multi_level_smoothing_score_in_space() {
        let problem = DummyProblem;
        let multi = MultiLevelSmoothing::new(5, 50);
        let solution = 5i32;

        // space_id=0: K=50, but only 3 neighbours available
        // score = (16 + 25 + 36) / 3 ≈ 25.667
        let score0 = multi.score_in_space(0, &problem, &solution);
        assert!((score0 - (77.0 / 3.0)).abs() < 1e-10);

        // space_id=4: K=1 (original)
        // score = 25.0
        let score4 = multi.score_in_space(4, &problem, &solution);
        assert!((score4 - 25.0).abs() < 1e-10);
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
