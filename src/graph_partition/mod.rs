mod descent;
mod graph;
mod state;

pub(crate) use descent::{BestImprovement, NonFinite};
pub use graph::Graph;
pub use state::PartitionState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    Flip(usize),
    Swap(usize, usize),
}
