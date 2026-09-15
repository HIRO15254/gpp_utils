mod graph;
mod state;

pub use graph::Graph;
pub use state::PartitionState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    Flip(usize),
    Swap(usize, usize),
}
