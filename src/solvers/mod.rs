//! 最適化ソルバーの実装。
//!
//! 新フレームワーク用の様々なソルバー（探索戦略）を提供。

mod engine;
pub(crate) use engine::{Engine, StepStatus};
