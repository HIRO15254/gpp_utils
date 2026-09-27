//! 最適化ソルバーの実装。
//!
//! 新フレームワーク用の様々なソルバー（探索戦略）を提供。

mod engine;
mod eo;
mod metropolis;
pub(crate) use engine::{Advance, Engine, StepStatus};
