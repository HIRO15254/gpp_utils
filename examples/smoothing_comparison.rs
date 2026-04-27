//! 新フレームワークの使用例。
//!
//! 複数のスムージング戦略で山登り法を実行し、結果を比較する。

use gpp_utils::graph_partition::{GraphGenerationMethod, GraphPartitionProblem};
use gpp_utils::optimization::{Problem, Solver};
use gpp_utils::smoothing::{AllNeighbourAveragingSmoothing, KAveragingSmoothing, NoSmoothing};
use gpp_utils::solvers::HillClimbingSolver;
use gpp_utils::experiment::BasinEvaluator;
use rand_mt::Mt19937GenRand64;

fn main() {
    println!("=== 新解最適化フレームワーク デモ ===\n");

    // 1. グラフを生成
    println!("📊 グラフを生成中...");
    let method = GraphGenerationMethod::Random {
        node_count: 20,
        expected_degree: 4.0,
    };
    let mut rng = Mt19937GenRand64::new(42);
    let problem = GraphPartitionProblem::generate(method, &mut rng);
    println!("   ✓ 20 個の頂点、期待次数 4.0\n");

    // 2. 初期解を生成
    let initial = problem.random_solution(&mut rng);
    let initial_score = problem.score(&initial);
    println!("🎯 初期解のスコア: {:.2}\n", initial_score);

    // 3. 複数のスムージング戦略で最適化
    println!("🔄 複数のスムージング戦略で最適化を実行中...\n");

    let solver = HillClimbingSolver::new();

    // 3a. スムージングなし
    {
        println!("【戦略 1】スムージングなし");
        let smoothing = NoSmoothing;
        let (_solution, stats) = solver.solve(&problem, &smoothing, initial.clone(), 42);

        println!("  最初のスコア:     {:.2}", stats.initial_score);
        println!("  最終スコア:       {:.2}", stats.final_score);
        println!("  最良スコア:       {:.2}", stats.best_score);
        println!("  ステップ数:       {}", stats.iterations_completed);

        // ベイスン評価
        let basin = BasinEvaluator::evaluate(&problem, &smoothing, &initial);
        println!("  初期解のベイスン: {:.2}\n", basin);
    }

    // 3b. K=5 近傍平均
    {
        println!("【戦略 2】K=5 近傍平均");
        let smoothing = KAveragingSmoothing::new(5);
        let (_solution, stats) = solver.solve(&problem, &smoothing, initial.clone(), 42);

        println!("  最初のスコア:     {:.2}", stats.initial_score);
        println!("  最終スコア:       {:.2}", stats.final_score);
        println!("  最良スコア:       {:.2}", stats.best_score);
        println!("  ステップ数:       {}", stats.iterations_completed);

        // ベイスン評価
        let basin = BasinEvaluator::evaluate(&problem, &smoothing, &initial);
        println!("  初期解のベイスン: {:.2}\n", basin);
    }

    // 3c. K=10 近傍平均
    {
        println!("【戦略 3】K=10 近傍平均");
        let smoothing = KAveragingSmoothing::new(10);
        let (_solution, stats) = solver.solve(&problem, &smoothing, initial.clone(), 42);

        println!("  最初のスコア:     {:.2}", stats.initial_score);
        println!("  最終スコア:       {:.2}", stats.final_score);
        println!("  最良スコア:       {:.2}", stats.best_score);
        println!("  ステップ数:       {}", stats.iterations_completed);

        // ベイスン評価
        let basin = BasinEvaluator::evaluate(&problem, &smoothing, &initial);
        println!("  初期解のベイスン: {:.2}\n", basin);
    }

    // 3d. 全近傍平均
    {
        println!("【戦略 4】全近傍平均（決定論的）");
        let smoothing = AllNeighbourAveragingSmoothing;
        let (_solution, stats) = solver.solve(&problem, &smoothing, initial.clone(), 42);

        println!("  最初のスコア:     {:.2}", stats.initial_score);
        println!("  最終スコア:       {:.2}", stats.final_score);
        println!("  最良スコア:       {:.2}", stats.best_score);
        println!("  ステップ数:       {}", stats.iterations_completed);

        // ベイスン評価
        let basin = BasinEvaluator::evaluate(&problem, &smoothing, &initial);
        println!("  初期解のベイスン: {:.2}\n", basin);
    }

    println!("✅ デモ完了！");
    println!("\n💡 異なるスムージング戦略で、探索の深さやステップ数が変わることが確認できます。");
}
