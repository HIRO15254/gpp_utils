//! グラフ仕様（プリセットされた N, D, 生成方式, シード）と
//! ファイルへの永続化機能を提供する。
//!
//! `data/graphs/` 以下に JSON 形式で保存され、再利用が可能。

use std::path::{Path, PathBuf};

use rand::Rng;
use rand_mt::Mt19937GenRand64;
use serde::{Deserialize, Serialize};

use crate::file_utils::{ensure_dir_exists, load_json, save_json};
use crate::graph_partition::{Graph, GraphPartitionProblem};

/// ノード数の選択肢。
pub const NODE_COUNTS: &[usize] = &[62, 124, 250, 500, 1000, 2000];

/// 期待次数の選択肢。
pub const EXPECTED_DEGREES: &[f64] = &[2.5, 5.0, 10.0, 20.0, 40.0];

/// グラフの生成方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphKind {
    Random,
    Geometric,
}

impl GraphKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Geometric => "geom",
        }
    }
}

/// グラフ仕様（識別子）。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GraphSpec {
    pub kind: GraphKind,
    pub n: usize,
    pub d: f64,
    pub seed: u64,
}

impl GraphSpec {
    /// 一意なファイル名（拡張子なし）。
    pub fn id(&self) -> String {
        let d_str = if (self.d.fract()).abs() < 1e-9 {
            format!("{}", self.d as i64)
        } else {
            format!("{:.1}", self.d).replace('.', "p")
        };
        format!("{}_n{}_d{}_s{}", self.kind.label(), self.n, d_str, self.seed)
    }

    /// 標準保存パス。
    pub fn file_path(&self, base_dir: &Path) -> PathBuf {
        base_dir.join(format!("{}.json", self.id()))
    }
}

/// 永続化されるグラフ。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredGraph {
    pub spec: GraphSpec,
    pub adjacency_list: Vec<Vec<usize>>,
    /// 幾何グラフの場合は座標を保存する。Random 生成では None。
    pub coordinates: Option<Vec<(f64, f64)>>,
    pub edge_count: usize,
}

impl StoredGraph {
    /// 仕様からグラフを生成する。
    pub fn generate(spec: GraphSpec) -> Self {
        let mut rng = Mt19937GenRand64::new(spec.seed);
        let (graph, coords) = match spec.kind {
            GraphKind::Random => (generate_random(spec.n, spec.d, &mut rng), None),
            GraphKind::Geometric => {
                let (g, c) = generate_geometric(spec.n, spec.d, &mut rng);
                (g, Some(c))
            }
        };
        let edge_count = graph.adjacency_list.iter().map(Vec::len).sum::<usize>() / 2;
        Self {
            spec,
            adjacency_list: graph.adjacency_list,
            coordinates: coords,
            edge_count,
        }
    }

    pub fn graph(&self) -> Graph {
        Graph {
            adjacency_list: self.adjacency_list.clone(),
            node_count: self.spec.n,
        }
    }

    pub fn problem(&self) -> GraphPartitionProblem {
        GraphPartitionProblem::new(self.graph())
    }

    /// 円形配置（Random グラフの可視化用）または保存された座標を返す。
    pub fn display_coords(&self) -> Vec<(f64, f64)> {
        if let Some(c) = &self.coordinates {
            return c.clone();
        }
        let n = self.spec.n;
        (0..n)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                (0.5 + 0.45 * a.cos(), 0.5 + 0.45 * a.sin())
            })
            .collect()
    }
}

/// グラフライブラリ（保存ディレクトリ）の管理。
pub struct GraphLibrary {
    pub base_dir: PathBuf,
}

impl GraphLibrary {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    pub fn ensure_dir(&self) -> std::io::Result<()> {
        ensure_dir_exists(&self.base_dir).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, format!("{}", e))
        })
    }

    pub fn path_for(&self, spec: &GraphSpec) -> PathBuf {
        spec.file_path(&self.base_dir)
    }

    pub fn exists(&self, spec: &GraphSpec) -> bool {
        self.path_for(spec).exists()
    }

    /// 仕様に対応するグラフをロードする（存在すれば）。
    pub fn load(&self, spec: &GraphSpec) -> Option<StoredGraph> {
        load_json::<StoredGraph>(&self.path_for(spec)).ok()
    }

    /// 仕様に対応するグラフを取得する。存在しなければ生成して保存する。
    pub fn load_or_generate(&self, spec: GraphSpec) -> Result<StoredGraph, String> {
        if let Some(g) = self.load(&spec) {
            return Ok(g);
        }
        self.ensure_dir().map_err(|e| format!("create dir: {}", e))?;
        let stored = StoredGraph::generate(spec);
        save_json(&stored, &self.path_for(&spec))
            .map_err(|e| format!("save: {}", e))?;
        Ok(stored)
    }

    /// 保存されているグラフを列挙する。
    pub fn list(&self) -> Vec<StoredGraph> {
        let mut v = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.base_dir) {
            for ent in rd.flatten() {
                let p = ent.path();
                if p.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Ok(g) = load_json::<StoredGraph>(&p) {
                        v.push(g);
                    }
                }
            }
        }
        v.sort_by(|a, b| {
            (a.spec.kind as u8, a.spec.n, a.spec.d as i64, a.spec.seed)
                .cmp(&(b.spec.kind as u8, b.spec.n, b.spec.d as i64, b.spec.seed))
        });
        v
    }
}

fn generate_random(n: usize, d: f64, rng: &mut Mt19937GenRand64) -> Graph {
    let mut g = Graph::new(n);
    let p = if n > 1 { d / (n - 1) as f64 } else { 0.0 };
    for i in 0..n {
        for j in (i + 1)..n {
            if rng.r#gen::<f64>() < p {
                g.add_edge(i, j);
            }
        }
    }
    g
}

fn generate_geometric(n: usize, d: f64, rng: &mut Mt19937GenRand64) -> (Graph, Vec<(f64, f64)>) {
    let mut g = Graph::new(n);
    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        pts.push((rng.r#gen::<f64>(), rng.r#gen::<f64>()));
    }
    let pi = std::f64::consts::PI;
    let threshold = (d / (n as f64 * pi)).sqrt();
    for u in 0..n {
        for v in (u + 1)..n {
            let dx = pts[u].0 - pts[v].0;
            let dy = pts[u].1 - pts[v].1;
            if (dx * dx + dy * dy).sqrt() <= threshold {
                g.add_edge(u, v);
            }
        }
    }
    (g, pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_format() {
        let s = GraphSpec { kind: GraphKind::Random, n: 124, d: 5.0, seed: 7 };
        assert_eq!(s.id(), "random_n124_d5_s7");
        let s2 = GraphSpec { kind: GraphKind::Geometric, n: 62, d: 2.5, seed: 0 };
        assert_eq!(s2.id(), "geom_n62_d2p5_s0");
    }

    #[test]
    fn test_generate_deterministic() {
        let s = GraphSpec { kind: GraphKind::Random, n: 30, d: 4.0, seed: 1 };
        let g1 = StoredGraph::generate(s);
        let g2 = StoredGraph::generate(s);
        assert_eq!(g1.adjacency_list, g2.adjacency_list);
    }
}
