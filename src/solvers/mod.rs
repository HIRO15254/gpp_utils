//! 最適化ソルバーの実装。
//!
//! 新フレームワーク用の様々なソルバー（探索戦略）を提供。

pub mod hill_climbing;

pub use hill_climbing::HillClimbingSolver;
