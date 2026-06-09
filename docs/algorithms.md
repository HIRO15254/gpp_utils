# アルゴリズム解説

`gpp_utils` が実装するグラフ分割問題と各メタヒューリスティクスについて、
**数学的背景**と**実装の詳細**の両面から解説する。コードを読む前の見取り図、
および研究ノートとして使うことを想定している。

> 記号の対応: 本書の数式記号と Rust 識別子の対応は各節で都度示す。
> ファイル参照は `src/` からの相対パスで表記する。

## 目次

1. [問題定義: グラフ分割問題](#1-問題定義-グラフ分割問題)
2. [差分評価 (Δ評価)](#2-差分評価-δ評価)
3. [グラフ生成](#3-グラフ生成)
4. [スムージング戦略](#4-スムージング戦略)
5. [ソルバー](#5-ソルバー)
6. [実験ワークフロー (run_executor)](#6-実験ワークフロー-run_executor)
7. [計算量まとめ](#7-計算量まとめ)
8. [再現性](#8-再現性)

---

## 1. 問題定義: グラフ分割問題

実装: `graph_partition.rs`

### 1.1 入力

無向・重みなしグラフ $G = (V, E)$ を隣接リストで表現する。
頂点数を $n = |V|$、辺数を $m = |E|$ とする。

```rust
pub struct Graph {
    pub adjacency_list: Vec<Vec<usize>>,  // adjacency_list[v] = v の隣接頂点
    pub node_count: usize,                // n
}
```

### 1.2 解の表現

解は各頂点を 2 群のいずれかに割り当てる写像であり、`Vec<bool>` で表す。

$$
x \in \{0, 1\}^n, \qquad x_v \in \{\texttt{false}, \texttt{true}\}
$$

```rust
pub type Partition = Vec<bool>;
```

群サイズを次のように定義する。

$$
t = |\{v : x_v = \texttt{true}\}|, \qquad f = n - t
$$

### 1.3 目的関数

最小化する目的関数 $S(x)$ は **カットエッジ数** と **バランスペナルティ** の和。

$$
S(x) \;=\; \underbrace{\bigl|\{(u,v) \in E : x_u \neq x_v\}\bigr|}_{\text{カット数 } C(x)}
\;+\; \underbrace{\alpha \,(t - f)^2}_{\text{バランスペナルティ}}
$$

- **カット数** $C(x)$: 端点が異なる群に属する辺の本数。分割の「悪さ」の主指標。
- **バランスペナルティ**: 群サイズ差の二乗。極端に偏った分割（例: 全頂点が片側）を抑制する。係数は定数 $\alpha = 0.05$（`ALPHA`）。

実装 (`GraphPartitionProblem::score`) は各辺を両端から 2 回数えて $2$ で割る:

```rust
for node in 0..n {
    for &neighbor in &adjacency_list[node] {
        if partition[node] != partition[neighbor] { cut_edges += 1; }
    }
}
cut_edges /= 2;                       // 各辺は 2 回数えられる
let diff = (t as i64 - f as i64).abs() as f64;
cut_edges as f64 + ALPHA * diff * diff
```

このとき $C(x)$ は整数、$(t-f)^2$ も整数なので、$S(x)$ は

$$
S(x) = \big(\text{整数}\big) + \alpha \cdot \big(\text{整数}\big)
$$

という形をとる。**この演算順序が後述のビット完全一致の鍵**になる。

### 1.4 近傍構造

近傍操作は **1 頂点のフリップ**（群割り当ての反転）のみ。

$$
N(x) = \{\, x^{(v)} : v \in V \,\}, \qquad
x^{(v)}_u = \begin{cases} \lnot x_u & (u = v) \\ x_u & (u \neq v) \end{cases}
$$

近傍サイズは常に $|N(x)| = n$。`neighbour(solution)` は $n$ 個の解を生成して
返すが、これは $O(n^2)$ のクローンを生むため、実用パスでは後述の差分評価を使う。

---

## 2. 差分評価 (Δ評価)

実装: `graph_partition.rs` の `delta_apply` / `delta_apply_cached` / `flip_vertex` / `compute_cuts_at`

近傍解のスコアを毎回 $O(n + m)$ で再計算するのは無駄が大きい。フリップは
局所操作なので、スコア変化は $O(\deg v)$ あるいは $O(1)$ で求められる。

### 2.1 カット数の差分

頂点 $v$ に接続する辺のうち、現在カットされている本数を

$$
c_v = \bigl|\{u : (u,v)\in E,\ x_u \neq x_v\}\bigr|
$$

とする。$v$ をフリップすると $v$ に接続する **全辺のカット状態が反転** する:

- カット中だった $c_v$ 本 → 非カットに
- 非カットだった $\deg(v) - c_v$ 本 → カットに

よってカット数の変化は

$$
\Delta C = \bigl(\deg(v) - c_v\bigr) - c_v = \deg(v) - 2c_v
$$

$$
\boxed{\;C(x^{(v)}) = C(x) + \deg(v) - 2c_v\;}
$$

### 2.2 バランスペナルティの差分

$v$ をフリップすると群サイズが $\pm 1$ ずつ動く:

$$
(t, f) \;\longrightarrow\;
\begin{cases}
(t-1,\ f+1) & (x_v = \texttt{true}) \\
(t+1,\ f-1) & (x_v = \texttt{false})
\end{cases}
$$

新しいサイズ差を $d' = |t' - f'|$ とすれば、新スコアは

$$
S(x^{(v)}) = C(x^{(v)}) + \alpha\, d'^2
$$

実装ではペナルティの「差分」ではなく **新しい整数状態 $(C', t', f')$ から
スコアを再構成** する。これにより `score()` と同じ演算経路を通る。

### 2.3 整数状態追跡と `cuts_at` キャッシュ

3 つの API がそれぞれ計算量と引き換えに前提を変える:

| API | 計算量 | 前提 |
|---|---|---|
| `delta_apply` | $O(\deg v)$ | 隣接リストを走査して $c_v$ を数える |
| `delta_apply_cached` | $O(1)$ | `cuts_at[v]` ($=c_v$) が事前計算済み |
| `flip_vertex` | $O(\deg v)$ | フリップを適用し `cuts_at` を増分更新 |

`compute_cuts_at(x)` は全頂点の $c_v$ を $O(n + m)$ で計算した配列 `cuts_at`
を返す。これを保持しておけば、各近傍候補の評価が `delta_apply_cached` で
$O(1)$ になる。

`flip_vertex` は実際にフリップを適用しつつ `cuts_at` を整合させる:

```rust
for &u in &adjacency_list[idx] {
    if partition[u] != bi { cuts_at[u] -= 1; }   // u はカット中だった → 非カットへ
    else                  { cuts_at[u] += 1; }   // u は非カットだった → カットへ
}
cuts_at[idx] = degree - cuts_at[idx];             // idx の全辺が反転
partition[idx] = !partition[idx];
```

`flip_vertex` は **対合**（同じ頂点で 2 回呼ぶと元に戻る）であり、これを
利用して焼きなまし法の「棄却」を `unflip` で実現する（[6.3 節](#63-高速化-specialized-sa)）。

### 2.4 ビット完全一致の保証

差分評価の戻り値 `new_score` は、整数 $(C', t', f')$ を `score()` と
**完全に同じ式・同じ演算順序** に渡して計算する:

```rust
// score() と delta_apply の両方が通る唯一の式
fn score_from_state(cut: i32, t: usize, f: usize) -> f64 {
    let diff = (t as i64 - f as i64).abs() as f64;
    cut as f64 + ALPHA * diff * diff
}
```

IEEE 754 の演算は決定論的なので、同じ入力整数に同じ演算を施せば
**f64 のビットパターンまで一致** する。`graph_partition.rs` のテスト群
（`test_delta_apply_bitwise_equality` ほか）が `to_bits()` 比較でこれを検証し、
`tests/regression.rs` が旧実装の baseline との完全一致を保証する。

---

## 3. グラフ生成

実装: `graph_partition.rs` の `generate_random_graph` / `generate_geometric_graph`

乱数は MT19937（`rand_mt::Mt19937GenRand64`、[8 節](#8-再現性)参照）。

### 3.1 Erdős–Rényi ランダムグラフ $G(n, p)$

期待次数 $d$ を与え、辺確率を

$$
p = \frac{d}{n - 1}
$$

とする（頂点 $v$ は他の $n-1$ 頂点それぞれと確率 $p$ で結ばれ、期待次数 $(n-1)p = d$）。
全頂点対 $(i, j),\ i < j$ を辞書順に走査し、独立に確率 $p$ で辺を張る。

### 3.2 幾何ランダムグラフ

$[0,1]^2$ 上に $n$ 個の点を一様ランダムに配置し、ユークリッド距離が
閾値 $r$ 以下の点対に辺を張る。閾値は期待次数 $d$ から逆算する。

1 頂点の近傍は半径 $r$ の円内に入る他頂点であり、その期待個数は
点密度 $n$ と円の面積 $\pi r^2$ の積で近似できる:

$$
d \approx n \cdot \pi r^2 \quad\Longrightarrow\quad
r = \sqrt{\frac{d}{n\pi}}
$$

幾何グラフでは頂点座標も保存され（`StoredGraph::coordinates`）、GUI で
そのまま可視化に使う。Random グラフは座標を持たないため、可視化時は
円周上に等間隔配置する（`display_coords`）。

---

## 4. スムージング戦略

実装: `smoothing.rs`（`Smoothing` トレイト実装群）

スムージングは、ソルバーが見るスコアランドスケープを「ならす」層。
局所最適の谷を浅くして探索を脱出しやすくする狙いがある。同じ問題に対し
スコア評価関数だけを差し替えられるよう、`Smoothing` トレイトで抽象化する。

```rust
pub trait Smoothing<S: Clone> {
    fn score(&self, problem: &dyn Problem<S>, solution: &S) -> f64;
}
```

解 $x$ の近傍スコア列を $\{S(x^{(0)}), \dots, S(x^{(n-1)})\}$ と書く。

### 4.1 NoSmoothing

平滑化なし。実スコアをそのまま返す。

$$
\tilde S(x) = S(x)
$$

### 4.2 KAveragingSmoothing

先頭 $K$ 個の近傍スコアの算術平均（決定論的）。$K' = \min(K, n)$ として

$$
\tilde S(x) = \frac{1}{K'} \sum_{i=0}^{K'-1} S(x^{(i)})
$$

近傍の「先頭 $K$ 個」は頂点インデックス順なので決定論的。$K=1$ で 1 近傍のみ、
$K=n$ で全近傍平均に一致する。

### 4.3 RandomKSmoothing

距離 1 近傍から $K$ 個を**ランダムサンプリング**して平均する確率的スムージング。
内部 RNG を持ち、呼ばれるたびに乱数列が進むため、同じ解でも値が揺らぐ。

- $K \le n$ のとき: 距離 1 近傍 $n$ 個から Fisher–Yates で非復元 $K$ 個を選び平均。
- $K > n$ のとき: 距離 1 近傍を全て使い、不足分 $K - n$ 個を **距離 2 近傍**
  （2 ステップ先の解）から補充する。距離 2 近傍は重複排除しつつ列挙する。

$$
\tilde S(x) = \operatorname{mean}\bigl(\,\{S(y) : y \in \mathcal{R}\}\,\bigr),
\qquad \mathcal{R} \subseteq N(x) \cup N^2(x),\ |\mathcal{R}| = \min(K,\ |N(x) \cup N^2(x)|)
$$

### 4.4 WeightedNeighbourSmoothing

**全近傍平均**と**実スコア**の線形ブレンド（決定論的）。重み $w = K'/n$（$K' = \min(K,n)$）として

$$
\tilde S(x) = w \cdot \underbrace{\frac{1}{n}\sum_{i=0}^{n-1} S(x^{(i)})}_{\text{全近傍平均}}
\;+\; (1 - w)\cdot S(x)
$$

- $K = 0$: $w = 0$ → `NoSmoothing` と等価
- $K = n$: $w = 1$ → 全近傍平均と等価
- $0 < K < n$: 両者の連続的な内挿。$K$ を連続パラメータとして扱える。

> **実験ワークフローでの指定**: `smoothing.rs` の汎用版は $K$ から $w = K'/n$ を導くが、
> 実験ワークフロー（`SmoothingSpec::WeightedAverage`）は重み $w \in [0,1]$ を**直接指定**する
> （グラフサイズ $n$ に依存しないため）。数式は同一で、パラメータの与え方のみが異なる。

### 4.5 補足: AllNeighbourAveragingSmoothing

全近傍スコアの単純平均。`WeightedNeighbourSmoothing` の $K = n$ と一致する。

> **2 系統の実装について**: `smoothing.rs` のこれらは汎用 `Smoothing` トレイト経由
> （[5 節](#5-ソルバー)のソルバーが使用）。実験ワークフロー（`run_executor.rs`）は
> 同じ数式を **整数状態上で再実装** した高速版を持つ（[6.3 節](#63-高速化-specialized-sa)）。
> 両者は数値的に一致するよう設計されている。

---

## 5. ソルバー

実装: `solvers/`（`Solver` トレイト実装群）

ソルバーは `Problem` と `Smoothing` の任意の組み合わせを受け取り最適化を実行する。
返り値は最良解と実行統計 `SolverStats`。

```rust
pub trait Solver {
    fn solve<S: Clone>(
        &self, problem: &dyn Problem<S>, smoothing: &dyn Smoothing<S>,
        initial: S, seed: u64,
    ) -> (S, SolverStats);
}
```

以下では現在解を $x$、スムージングスコアを $\tilde S$ と書く。

### 5.1 山登り法 (Hill Climbing)

実装: `solvers/hill_climbing.rs`

**貪欲な局所探索**。各反復で全 $n$ 近傍を評価し、$\tilde S$ が最小の近傍へ移動する。
改善する近傍が無くなったら停止する（局所最適）。

$$
x \leftarrow \arg\min_{y \in N(x)} \tilde S(y)
\qquad \text{ただし } \min_{y} \tilde S(y) < \tilde S(x) \text{ の間のみ}
$$

- 単調減少なので必ず有限ステップで停止する。
- 到達点はスムージング空間 $\tilde S$ の局所最適。`NoSmoothing` なら実スコアの
  局所最適（= ベイスン）になる。
- **同スコアのタイブレーク**: $\tilde S$ が最小の近傍が複数あるときは、
  reservoir sampling で 1 つを一様ランダムに選ぶ。最小インデックス固定だと
  探索経路に系統的な偏りが出るため。乱数は `seed` 由来の専用列を使う。

### 5.2 焼きなまし法 (Simulated Annealing)

実装: `solvers/simulated_annealing.rs`

**固定温度** $T$ のメトロポリス法。各反復でランダムに 1 近傍 $y$ を選び、
スコア差 $\Delta = \tilde S(y) - \tilde S(x)$ に応じて受理判定する。

$$
P(\text{accept}) =
\begin{cases}
1 & (\Delta < 0) \\[4pt]
\exp\!\left(-\dfrac{\Delta}{T}\right) & (\Delta \ge 0,\ T > 0) \\[8pt]
0 & (\Delta \ge 0,\ T = 0)
\end{cases}
$$

- 改善は常に受理。悪化は温度 $T$ が高いほど受理されやすい（山越え）。
- $T = 0$ は悪化を一切受理しない貪欲法（ランダムウォーク版の山登り）。
- 本実装は温度を下げない（定数温度）。冷却は実験ワークフロー側で温度の
  異なる複数 `RunConfig` を用意して比較する設計（[6 節](#6-実験ワークフロー-run_executor)）。

温度はパラメータ $\Theta = \log_{10} T$ で与える（`RunConfig::theta`）。
$T = 10^\Theta$ なので $\Theta$ を等間隔に振ると温度が対数スケールで並ぶ。

#### 5.2.1 スワップ近傍版（`run_sa_swap` / `SolverSpec::SaSwap`）

フリップ近傍の SA とは別に、**スワップ近傍版 EO（[5.3 節](#53-extremal-optimization-τ-eo)）と
同一の近傍・厳密バランス・初期化・ベイスン記録を共有する SA** を併設している。違いは
受理規則だけ:

- **近傍 = スワップ**（$v_1\in A \leftrightarrow v_2\in B$）、**厳密バランス** $|A|=|B|=N/2$、
  初期解は `balanced_init`。ペナルティ項は一定（偶数 $N$ なら 0）なので実スコア = カット数。
- **1 手**: ランダムに 1 スワップを提案し、**メトロポリス基準**（温度 $\Theta=\log_{10}T$）で受理。
  $T=0$（`theta=None`）は改善スワップのみ受理する貪欲スワップ降下。
- `basin_*` は EO スワップ版（`run_eo`）と同じく **スワップ近傍の本物の局所最適**
  （`hill_climb_swap_fast`、厳密バランスを保つスワップ最急降下、$O(N^2)$/降下ステップ）。
  `final_partition = S_{\text{best}}$、smoothed == real、$m_{\text{best}}$ は `current_real` の累積最小から。

これにより **同一のスワップ近傍上で EO（ランク選択＋無条件受理）と SA（ランダム＋メトロポリス）を
直接比較**できる。フリップ近傍では SA ↔ EoFlip（[5.3.1 節](#531-フリップ近傍版-run_eo_flip--solverspeceoflip)）、
スワップ近傍では SaSwap ↔ Eo、という 2×2 の対応になる。

### 5.3 Extremal Optimization (τ-EO)

実装: 実験ワークフローの忠実版は `run_executor.rs` の `run_eo`
（[6 節](#6-実験ワークフロー-run_executor)経由で `execute` から呼ばれる）。
`solvers/extremal_optimization.rs` は汎用 `Solver` トレイト版で、二集合・スワップの
概念を持たない簡易版（ワークフローでは使わない）。

出典: Boettcher & Percus, *Phys. Rev. E* **64**, 026114 (2001)。
グラフ二分割に特化した τ-EO で、**バランス制約 $|A| = |B| = N/2$ を全ステップで
厳密維持**する（緩和しない）。各頂点に**適応度**を与え、適応度の低い（＝誤った集合に
いる疑いが濃い）頂点をべき乗則確率で選び、反対集合の頂点とスワップする。

**適応度（次数正規化）**: 頂点 $i$ について、同じ集合の隣接数 $g_i$、反対集合の隣接数
$b_i$、次数 $\deg_i = g_i + b_i$ とすると

$$
\lambda_i = \frac{g_i}{\deg_i} = \frac{\deg_i - b_i}{\deg_i}
$$

$b_i$ は既存の `cuts_at[i]`（$i$ に接続する横断辺数）そのものなので、`cuts_at` から
$O(1)$ で計算できる。孤立頂点（$\deg_i = 0$）は $\lambda_i = 1.0$ に固定する。
$\lambda_i$ が**小さいほど「悪い」**。

**統一ランクでのべき乗則選択**: 全 $N$ 頂点を $\lambda$ 昇順にランク付けし
（ランク 1 = 最悪、ランク $N$ = 最良）、ランク $k$ を確率

$$
P(k) \propto k^{-\tau}, \qquad 1 \le k \le N
$$

で引く。累積分布 $\mathrm{CDF}(k) = \dfrac{\sum_{j=1}^{k} j^{-\tau}}{\sum_{j=1}^{N} j^{-\tau}}$
を事前計算し、$u \sim \mathrm{Uniform}(0,1)$ を二分探索してランクを決める。
引いたランクの頂点 $v_1$ を取り、**$v_1$ と反対集合の頂点が当たるまで $k_2$ を引き直して**
$v_2$ を得る（再抽選の上限を超えたら反対集合から一様ランダムにフォールバック）。
$v_1$ と $v_2$ を**スワップ**する（＝ 2 連続フリップ）。バランスが厳密維持されるため、
スワップ後も $|A| = |B|$。

**無条件受理**: カットの増減にかかわらず常にスワップを適用する（SA のような受理判定なし）。
現在解は悪化しうるので、**最良解 $S_{\text{best}}$ は別途保存**し、それを最終解として返す。

指数 $\tau$ の既定値は **1.4**（実用範囲 $1.3 \sim 1.6$）。$\tau$ は普遍定数ではなく
（問題・サイズ・実行時間依存）、`RunConfig` の `SolverSpec::Eo { tau }` で設定可能。
$\tau \to 0$ でほぼ一様（ランダムウォーク、収束しない）、$\tau \to \infty$ で常に最悪を
選ぶ貪欲法（即 jam）。最適はその中間（"ergodic edge"）。

**バランス時のスコア = カット数**: 偶数 $N$ なら $|V_1| = |V_2| = N/2$ でペナルティ項
$\alpha(|V_1|-|V_2|)^2 = 0$ となり、実スコア $= $ カット数。奇数 $N$ は
$|V_1| = \lceil N/2 \rceil,\ |V_2| = \lfloor N/2 \rfloor$ 固定でペナルティは一定
（$\alpha\cdot 1$）。いずれもスコア最小化 ≡ カット最小化。

**計算量**: フェーズ1（正しさ優先）は毎ステップ全頂点を $\lambda$ でソートするため
$O(N \log N)$/反復。フェーズ2では順序統計木/ヒープで $\lambda$ を保持し、スワップで
変化する $v_1, v_2$ とその隣接頂点の $\lambda$ のみ局所更新すれば $O(\deg \log N)$/反復に
できる（コード内に `// PHASE 2:` で置換余地を明示）。

#### 5.3.1 フリップ近傍版（`run_eo_flip` / `SolverSpec::EoFlip`）

スワップ版（厳密バランス）とは別に、**SA と同一の近傍・目的関数・ベイスン算出を共有する
フリップ近傍版**を併設している。違いは「1 手の選び方」だけ:

- **近傍 = 単一フリップ**、**目的関数 = SA と同じ** $\text{cut} + \alpha(|V_1|-|V_2|)^2$、
  初期解は `random_solution`（SA と同じ）。バランスは厳密制約ではなくペナルティ項で扱う。
- **適応度**: $g/\deg$ にバランスペナルティを「悪い辺 / 良い辺」として織り込む対称版。
  頂点 $i$ をフリップしたときの符号付き不均衡 $\text{diff}=t-f$ の変化を $\text{diff}'$ とし、
  $\text{improvement}_i = \alpha(\text{diff}^2 - \text{diff}'^2)$、$q=|\text{improvement}_i|$ とすると:

$$
\lambda^{\text{eff}}_i =
\begin{cases}
\dfrac{g_i}{\deg_i + q} & (\text{improvement}_i > 0,\ \text{多数派側 → 悪い辺と deg に } q) \\[8pt]
\dfrac{g_i + q}{\deg_i + q} & (\text{improvement}_i < 0,\ \text{少数派側 → 良い辺と deg に } q) \\[8pt]
\dfrac{g_i}{\deg_i} & (\text{improvement}_i = 0)
\end{cases}
$$

  多数派頂点は $\lambda^{\text{eff}}$ が下がって選ばれやすく（不均衡を是正）、少数派頂点は
  上がって守られる。$\lambda^{\text{eff}}\in[0,1]$ を保ち暴走しない。孤立頂点は $\deg_i=0$ で
  多数派側なら $\lambda^{\text{eff}}=0/(0+q)=0$（カット0コストの自由な是正フリップとして最優先）、
  $q=0$ なら $1.0$。
- **選択・受理**: スワップ版と同じく $\lambda^{\text{eff}}$ 昇順ランクをべき乗則 $P(k)\propto k^{-\tau}$ で
  引き、選ばれた頂点を**無条件にフリップ**。最良解 $S_{\text{best}}$ を別途保存して返す。
- **ベイスン**: `make_snapshot_fast`（`no_smoothing=true`）＋ `hill_climb_real_fast` ＋
  タイブレーク RNG（`seed ^ TIEBREAK_SALT`）を **SA の `None` ケースと同一に**使う。
  よって `StepRecord` の 6 トレースは SA と1対1で比較でき、`basin_*` は（m_best ではなく）
  **本物の単一フリップ局所最適**になる（`current_*` は揺らぐ現在解、smoothed==real）。

用途: スワップ版は論文に忠実な厳密バランス比較用、フリップ版は **SA との直接比較
（同一ベイスン枠組み）** 用。`τ` の役割は両者共通。

### 5.4 Simulated Quantum Annealing (SQA)

実装: `solvers/simulated_quantum_annealing.rs`

横磁場イジング模型の量子焼きなましを、**鈴木–トロッター分解**で古典シミュレート
する。量子系を $P$ 枚の古典レプリカ（トロッタースライス）のリングに写像し、
レプリカ間に強磁性結合を入れることで量子トンネル効果を模擬する。

**横磁場の減衰**: 横磁場 $\Gamma$ をステップ進行度 $s = \text{step}/\text{max\_steps}$
に対して指数的に減衰させる。

$$
\Gamma(s) = \Gamma_{\text{init}} \left(\frac{\Gamma_{\text{final}}}{\Gamma_{\text{init}}}\right)^{s}
$$

**レプリカ間結合**: 熱的温度 $T$、レプリカ数 $P$ に対し、隣接スライス間の
強磁性結合強度は

$$
J_\perp = -\frac{PT}{2}\,\ln\!\Bigl(\tanh\frac{\Gamma}{PT}\Bigr)
$$

$\Gamma$ が大きい（焼きなまし初期）ほど $J_\perp$ は弱く、レプリカは独立に
動ける。$\Gamma \to 0$ で $J_\perp \to \infty$ となり全レプリカが揃う。

**1 ステップ**: $P \times n$ 回のフリップ試行を行う。レプリカ $k$ のビット $i$ を
フリップする際のエネルギー変化は、問題由来項と結合項の和:

$$
\Delta E = \underbrace{\frac{S(\text{replica}_k^{(i)}) - S(\text{replica}_k)}{P}}_{\Delta E_{\text{problem}}}
\;+\; \underbrace{\Delta E_{\text{coupling}}}_{\text{隣接 2 スライスとの一致変化}}
$$

結合項は、フリップ前後で前後のスライスとの一致／不一致が変わるたびに
$\pm 2 J_\perp$ 寄与する。受理は温度 $T$ のメトロポリス基準
（$\exp(-\Delta E / T)$）。

> SQA はレプリカ間結合のため `Vec<bool>` 解を直接操作する。汎用 `Solver`
> トレイトには乗らず、専用の `solve()` メソッドを持つ。

---

## 6. 実験ワークフロー (run_executor)

実装: `run_executor.rs`、設定は `graph_spec.rs` / `run_config.rs`

GUI（`bin/gui.rs`）が回す実験のバックボーン。**プリセット条件で大量の探索を
実行し、対数刻みでスナップショットを取ってファイルにキャッシュする**。

`execute()` は `RunConfig::solver`（`SolverSpec`）でソルバーを分岐する:

- `SolverSpec::Sa`（既定）: `cfg.smoothing` に応じて
  `run_sa_none` / `run_sa_kavg` / `run_sa_random_k` / `run_sa_weighted` を呼ぶ。
- `SolverSpec::SaSwap`: スワップ近傍・厳密バランス版 SA `run_sa_swap` を呼ぶ
  （[5.2.1 節](#521-スワップ近傍版-run_sa_swap--solverspecsaswap)）。
- `SolverSpec::Eo { tau }`: 厳密バランスのスワップ版 τ-EO `run_eo` を呼ぶ
  （[5.3 節](#53-extremal-optimization-τ-eo)）。
- `SolverSpec::EoFlip { tau }`: フリップ近傍版 τ-EO `run_eo_flip` を呼ぶ
  （[5.3.1 節](#531-フリップ近傍版-run_eo_flip--solverspeceoflip)）。SA と同一のベイスン算出を共有する。

ここで使う SA は [5.2 節](#52-焼きなまし法-simulated-annealing)の汎用ソルバーとは
**別実装**で、`GraphPartitionProblem` に特化して差分評価で高速化したもの。
数値結果は汎用版とビット完全一致する（[2.4 節](#24-ビット完全一致の保証)）。
`run_eo` も同じ整数状態（`cut`, `t`, `f`）と `cuts_at` / `flip_vertex` /
`delta_apply_cached` を再利用してスワップを差分評価する。

### 6.1 対数刻みスナップショット

ステップ $1, 2, \dots, 9, 10, 20, \dots, 90, 100, 200, \dots$ という対数刻みで
スナップショットを取る（`logarithmic_steps`）。末尾が `max_iter` に一致しない
場合は `max_iter` を追加する。各 10 進の桁で 9 点ずつサンプリングするので、
$10^N$ 反復でも記録数は $O(9N)$ に収まる。

反復数は $10^N$（`RunConfig::log10_iterations`、最大 $10^9$）で指定する。

### 6.2 6 トレースの定義

各スナップショットは 1 つの `StepRecord` を記録する。中心となるのは
**現在解** と、そこから 2 種類の空間で山登りした **ベイスン**（局所最適解）。

| フィールド | 内容 |
|---|---|
| `current_smoothed` | 現在解のスムージング空間スコア $\tilde S(x)$ |
| `current_real` | 現在解の実スコア $S(x)$ |
| `basin_smoothed_from_smoothed` | $\tilde S$ で山登り → 到達点の $\tilde S$ |
| `basin_real_from_smoothed` | $\tilde S$ で山登り → 到達点の $S$ |
| `basin_smoothed_from_real` | $S$ で山登り → 到達点の $\tilde S$ |
| `basin_real_from_real` | $S$ で山登り → 到達点の $S$ |

「どの空間で山登りすると、どちらの空間で深い谷に着くか」を比較するための
6 トレース。スムージングなし（`None`）の場合はスムージング空間 = 実空間なので
4 フィールドが同値となり、山登りは 1 回で済む。

**EO / SaSwap のトレース解釈（プレーン）**: これらは平滑化を行わないので smoothed = real。
`current_smoothed = current_real = その時点の現在解の生スコア`（無条件受理／メトロポリスで
揺らぐ軌跡）、`final_partition = S_best`。`basin_*`（4 フィールド同値）は **その近傍での本物の
局所最適（ベイスン）**:

- **フリップ近傍（`Sa` / `run_eo_flip`）**: `hill_climb_real_fast`（単一フリップ最急降下）。
- **スワップ近傍（`run_sa_swap` / `run_eo`）**: `hill_climb_swap_fast`（厳密バランスを保つ
  スワップ最急降下、$O(N^2)$/降下ステップ）。

いずれもタイブレーク RNG（`seed ^ TIEBREAK_SALT`）を使い、本体 RNG とは独立なので
`final_partition` には影響しない。同一近傍のソルバ同士（`Sa`↔`EoFlip`、`SaSwap`↔`Eo`）は
ベイスン算出が一致するので 6 トレースを 1対1 で比較できる。べき則検証に使う $m_{\text{best}}$ は
記録の `current_real` の累積最小から得る。

ベイスン計算の山登りも [5.1 節](#51-山登り法-hill-climbing)と同様、同スコア近傍を
一様ランダムにタイブレークする。タイブレーク用の乱数列は SA 本体（`rng`）とは
独立した専用列（`seed ^ TIEBREAK_SALT`）なので、`rng` の消費には影響しない。
ただし確率的スムージング `RandomKAverage` では、ベイスン山登りがスムージングを
評価する際にスムージング用乱数列 `sm_rng` を消費する。タイブレークの結果で
山登り経路が変わると `sm_rng` の進み方も変わり、後続の SA 反復に波及して
`final_partition` も変化しうる（決定論的な `None` / `KAverage` /
`WeightedAverage` では `sm_rng` を使わないためこの波及は起きない）。

### 6.3 高速化: specialized SA

SA ループ本体（`run_sa_generic`）は整数状態 $(C, t, f)$ と配列 `cuts_at` を
ループ全体で保持し、汎用版の $O(n^2)$ クローンを完全に排除する。

1 反復の流れ:

```text
idx = random(0..n)
(C', t', f') = delta_apply_cached(...)   # O(1): 候補の整数状態
flip_vertex(idx)                          # O(deg): current と cuts_at を候補へ
neighbour_smoothed = sm(...)               # スムージング評価
if accept:  状態を確定 (C,t,f) ← (C',t',f')
else:       flip_vertex(idx) で unflip     # 対合性を利用して原状復帰
```

スムージング評価 `sm` も整数状態経由で行う specialized 版を smoothing 種別ごとに
持つ（`run_sa_none` / `run_sa_kavg` / `run_sa_random_k` / `run_sa_weighted`）。
これらは [4 節](#4-スムージング戦略)の各戦略を整数状態上で再実装したもので、
**RNG 消費順・浮動小数点演算順を元実装と一致させる**ことでビット完全一致を保つ。

並列化: GUI は `(graph, config, seed)` の三つ組ジョブを rayon で並列実行し、
キャッシュ済みの三つ組はスキップする。シードごとに独立な乱数列なので
並列実行しても結果は逐次実行とバイト単位で一致する。

---

## 7. 計算量まとめ

$n$ = 頂点数、$m$ = 辺数、$\bar d = 2m/n$ = 平均次数、$K$ = スムージングのサンプル数。

| 操作 | 計算量 |
|---|---|
| `score`（全再計算） | $O(n + m)$ |
| `delta_apply` | $O(\deg v)$ |
| `delta_apply_cached` | $O(1)$ |
| `flip_vertex`（`cuts_at` 増分更新込み） | $O(\deg v)$ |
| `compute_cuts_at` | $O(n + m)$ |
| スムージング `None` の 1 評価 | $O(1)$ |
| スムージング `KAverage` の 1 評価 | $O(K)$ |
| スムージング `WeightedAverage` の 1 評価 | $O(n)$ |
| specialized SA の 1 反復（`None`） | $O(\bar d)$ |
| 山登り 1 ステップ（実空間、`cuts_at` 利用） | $O(n)$ |
| 山登り 1 ステップ（スムージング空間） | $O(n \cdot \text{sm 評価コスト})$ |

---

## 8. 再現性

すべての乱数は **メルセンヌ・ツイスタ MT19937**（`rand_mt::Mt19937GenRand64`）。
同一シードからは同一の乱数列が得られるため、本ライブラリの計算は完全に決定論的。

- **グラフ生成**: `GraphSpec::seed` から生成。同じ `(kind, n, d, seed)` は
  常に同じグラフを生む（`data/graphs/<id>.json` にキャッシュ）。
- **SA 実行**: `execute(..., seed)` の `seed` が SA ループの乱数列を決める。
  確率的スムージング `RandomKAverage` は別系列
  `sm_seed = seed + 0xDEAD_BEEF` を使い、SA 本体の乱数列とは別ストリームになる。
- **ベイスン山登りのタイブレーク**: 同スコア近傍の一様選択には専用乱数列
  `seed ^ TIEBREAK_SALT` を使う。SA 本体の `rng` とは独立。`RandomKAverage`
  のときのみ、ベイスン山登りが `sm_rng` を消費する経路を通じて間接的に
  SA 軌跡へ影響する（[6.2 節](#62-6-トレースの定義)参照）。
- **並列実行**: ジョブ（シード）ごとに乱数列が独立なので、並列実行しても
  逐次実行とバイト単位で一致する。

数値結果のビット完全一致は次の 2 段で担保される:

1. 差分評価が `score()` と同じ式（`score_from_state`）を通る（[2.4 節](#24-ビット完全一致の保証)）。
2. `tests/regression.rs` が旧実装で採取した baseline と
   `final_partition` のビット一致・`records` の $10^{-12}$ 一致を検証する。

---

## 参考: モジュールとアルゴリズムの対応

| モジュール | 担当 |
|---|---|
| `optimization.rs` | `Problem` / `Smoothing` / `Solver` トレイト定義 |
| `graph_partition.rs` | 問題定義・目的関数・差分評価（2 節） |
| `smoothing.rs` | スムージング戦略（4 節） |
| `solvers/` | 4 ソルバー（5 節） |
| `experiment.rs` | ベイスン評価 `BasinEvaluator`（汎用 `Smoothing` 経由の山登り） |
| `graph_spec.rs` | グラフ生成（3 節）と永続化 |
| `run_config.rs` | SA 実行条件（温度 $\Theta$、反復数、スムージング種別） |
| `run_executor.rs` | 実験ワークフロー・specialized SA（6 節） |
| `bin/gui.rs` | 実験 GUI |
