# 定温SA（MA）のための解空間平滑化：先行研究の調査と新規手法の提案

本書は調査と提案の記録であり、計算仕様の正本ではない。計算規則の正本は[algorithms.md](algorithms.md)、実験条件の正本は[AGENTS.md](../AGENTS.md)の「実験のベースライン」である。本書の提案は未実装で、`gpp`の挙動を変えない。第5章の予備実験は`gpp`の外で作ったプロトタイプによるもので、ベースラインの正式な比較ではない。再現用のスクリプトは[experiments_sa_eo/smoothing_survey/](../experiments_sa_eo/smoothing_survey/README.md)にある。

作成日：2026-09-26

## 要旨

- **対象**：GPP（`cut + alpha*(|A|-|B|)^2`、alpha=0.05、Flip／Swap近傍）に対する定温SA（Metropolis法。以下MA）に使える解空間平滑化。
- **既存実装の整理（第3章）**
  - 全近傍平均（`all_average`）と重み付き平均（`weighted_average`）は、FlipでもSwapでも元の目的関数の一次関数になる。MAでは温度を付け替えただけの探索と同じである。Flipの結果は復旧run計画で既知であり、本書でSwapの係数 $1-8/n+8/n^2$ を加えた。距離2の近傍平均、独立なランダム反転の期待値など、近傍構造から作る線形の平均はすべて同じ結論になる（elementary landscapeの帰結）。
  - 重みなしグラフにGu–Huang型のインスタンス平滑化を当てると、温度とalphaの付け替えに退化する。Swapでは温度だけの付け替えになる。
  - `random_k_average`は、評価値を次の受理まで保持するため、pseudo-marginal MHになる。定常分布は $\pi_K(s)\propto \mathbb{E}_S[\exp(-\sum_{w\in S}H(s^{(w)})/(KT))]$ である。
    - K=1では、現行解から保持中の1頂点を反転した「影の解」が温度Tのボルツマン分布に厳密に従う。目標分布は平滑化されず、提案が実質3頂点反転になり、観測に1頂点分のずれが乗るだけである。
    - K≥2では、軟らかさKTの近傍ソフトミンに相当する非自明な平滑化になる。K→nで全近傍平均（温度の付け替え）に戻る。
    - 以上を小規模グラフの全列挙と長時間シミュレーションで確認した。
- **先行研究（第2章）**：探索空間平滑化（TSP起源）、ノイズ法、エネルギーの単調変換、履歴依存の地形変更、局所エントロピー、pseudo-marginal MCMC、粗視化、スペクトル法、定温MAの理論を整理した。次の3点は、調査の範囲では先行例が見つからなかった。
  - GPPへのGu–Huang型平滑化の適用と、重みなしデータでの退化の指摘
  - `random_k`型評価をpseudo-marginal連鎖として解析したもの
  - 以下の提案A〜Cの手法
- **提案（第4章）**
  - **A. 近傍ソフトミン平滑化（NSM）**：$F_{\lambda,\kappa}(s)=-\lambda\log(e^{-H(s)/\lambda}+\kappa\sum_{y\in N(s)}e^{-H(y)/\lambda})$。`random_k`の決定的で雑音のない一般化であり、利得バケットで1ステップ $O(\deg)$ で厳密に計算できる。λ→0で近傍最小値（エロージョン）になり、ハミング距離3以内の局所最小解の間の障壁が消える。
  - **B. 熱核スペクトル平滑化（HKS）**：辺重みを熱核 $e^{-tL}$ の非対角成分に置き換えたカットを使う。t→0で元の問題に一致し、t→∞で一様重みの平坦な地形に近づく。そのときの主な起伏はスペクトル二分割の方向を向く。重みなしグラフでも温度と等価にならないインスタンス平滑化である。
  - **C. ラベル対称なレプリカ結合MA**：局所エントロピーのGPP版で、結合項に重なりの2乗を使う。
  - **D. データ駆動のエネルギー依存温度**：単調変換であり、平滑化の効果を焼きなまし効果から切り分ける対照にも使う。
- **予備実験（第5章）**：ベースラインと同じ18グラフ・10^6ステップで、シード8個・Θ23点のプロトタイプ試走を行った。結果の要約は第5章に記す。
- **実験計画（第6章）**：ベースライン条件をそのまま使い、新しい軸は平滑化パラメータだけにする。まずシード0〜7でふるい分け、有望な系列だけ0〜31へ広げる。シードを後から足しても条件IDは変わらないため、結果を再利用できる。

## 1. 対象と前提

### 1.1 問題と記法

- グラフ $G=(V,E)$、$n=|V|$、$m=|E|$、頂点vの次数 $d_v$、平均次数 $\bar d=2m/n$。
- 解 $s\in\{+1,-1\}^n$（+1を群Aとする）、$M(s)=\sum_i s_i=|A|-|B|$。
- 目的関数 $H(s)=\mathrm{cut}(s)+\alpha M(s)^2$、$\alpha=0.05$。Johnsonら[1]がSAの実験に用いた罰金形と同じ形である。
- Flip近傍 $N_F(s)=\{s^{(v)}\}$ は頂点vの反転で、近傍数はnである。Swap近傍 $N_S(s)$ は等分割上で両群から1頂点ずつ交換するもので、近傍数は $(n/2)^2$ である。
- Flipの利得は $g_v(s)=H(s^{(v)})-H(s)=(d_v-2c_v)+4\alpha(1-s_vM)$ である。$c_v$ はvに接するカット辺の数。
- MA(T)：近傍から一様に候補を選び、探索評価Eの差Δに対して確率 $\min(1,e^{-\Delta/T})$ で受理する（`gpp`の`sa`）。平滑化とは探索評価をHから別の関数に置き換えることであり、`gpp`の設計と同じ扱いである。

### 1.2 定温MAから見た平滑化の類型

MAは温度を固定するので、平滑化の効果は、それが作るマルコフ連鎖だけで決まる。本書では次の類型を使う。

| 類型 | 探索評価 | MAでの意味 | 局所最小解の集合 |
|---|---|---|---|
| 温度等価 | $aH+b$（a>0） | 温度T/aのMAと同じ連鎖 | 変わらない |
| 単調変換 | $\varphi(H)$（φは増加関数） | 状態に依存する実効温度 $T/\varphi'(H)$ | 変わらない |
| 構造変更 | インスタンスや近傍の値に依存 | 目標分布と障壁が変わる | 変わる |
| 履歴依存 | 訪問履歴で更新 | 非定常 | 時間とともに変わる |
| 確率的評価 | 評価に乱数を含む | 目標は $\mathbb{E}[e^{-\hat E/T}]$ に比例 | 変わりうる |

Θを81点走査するベースラインでは、温度等価な平滑化から新しい情報は得られない。平滑化の強さを時間とともに変える場合は、実効温度の変化（隠れた焼きなまし）が混ざりうる。そのため、温度だけを同じように変える対照と比べる必要がある。

### 1.3 評価の観点

`gpp`は各計測点で、現行解と暫定解から実目的関数で山登りした値（ベイスン値）を記録する。平滑化した地形上の現行解はHの局所最小解から少しずれうるが、ベイスン値はこのずれを吸収する。本書でも暫定解のベイスン値を主な指標とする。

## 2. 先行研究

文献の書誌は、調査用サブエージェントがWeb検索結果の要約で確認したものである。この環境からは原典のPDFを開けなかった。未照合の項目は第8章に明記した。

### 2.1 探索空間平滑化（インスタンス平滑化）

- **Gu & Huang [2]**：TSPの距離（[0,1]に正規化）を平均 $\bar d$ へ縮める。
  - $d_{ij}\ge\bar d$ のとき $d_{ij}(a)=\bar d+(d_{ij}-\bar d)^a$、$d_{ij}<\bar d$ のとき $d_{ij}(a)=\bar d-(\bar d-d_{ij})^a$ とする（a≥1）。
  - aを大きな値から1へ下げながら、各段で局所探索を行う。a→∞では全巡回路が同じ長さになる自明な問題になる。
- **Schneiderら [3]**：線形・指数・双曲・シグモイド・対数の平滑化関数と、SAやGreat Deluge法との組み合わせをTSPで調べた。
- **Coyら [5]**：8種の平滑化を2-opt／3-optと組み合わせて比べた（TSP）。凸型と凹型の平滑化を交互に使う逐次平滑化（SSA）が最もよかった。同じグループの[4]は、fine-tuned learningと呼ぶ手法をTSPに適用した（内容の詳細は未照合）。
- **Schneider & Kirkpatrick [6]**：教科書 *Stochastic Optimization* に、地形を変える手法の章がある。
- **日本語文献**
  - 久保・Pedroso [7]の2.8節「探索空間平滑化法・交互平滑化法」。
  - 日本機械学会最適化シンポジウム2012の講演 [8]「メトロポリスアルゴリズムとの併用による探索空間平滑化法の機能特性 I・II」。MAと探索空間平滑化の組み合わせをTSPで調べ、平滑化係数の有効値、適応的に平滑化を戻す方法、ベイスン間の遷移を報告している（要約情報による。著者名は未確認）。定温MAと平滑化を組み合わせる点で、本研究の直接の先行研究である。
- **近年の研究**
  - Sunら [9]：既知の局所最適解から作る凸包TSPとの凸結合（TSP）。
  - [10,11]：元の問題と「おもちゃ問題」の凸結合 $g=(1-\lambda)f+\lambda\alpha\hat f$（UBQPとTSP）。[11]では、最も平坦なおもちゃ問題が最良ではなかった。
  - [12]：スピングラスの結合行列のべき乗による地形変換。
- **GPPへの含意**
  - Gu–Huang型は、インスタンスの重みのばらつきを一律に縮める。重みなしグラフの重みは0と1の2値なので、命題2のとおり温度とalphaの付け替えに退化する。
  - 凸結合型 [10,11] は、おもちゃ問題の選び方に強く依存する。GPPでの自然な選び方は自明でない。
  - 結合行列のスペクトル変換 [12] は、提案Bに最も近い既存手法である。

### 2.2 ノイズ法とデータ摂動

- **Charon & Hudry [13,14]**：データまたは評価差に雑音を加えて局所探索し、雑音を0へ減らしていく。[14]はSAや閾値受理法がノイズ法の特殊例として書けることを示した。適用先はクリーク分割やTSPなどである。
- **Sudhakar & Siva Ram Murthy [15]**：GPP（VLSI分割）への修正ノイズ法。調査で見つかった、GPPにデータ摂動を直接使った唯一の例である。
- 問題空間探索（Storerら [16]）、TSPの座標摂動（Codenottiら [17]）もデータを摂動する手法である。
- **含意**：評価差に加える雑音は、MAの受理乱数と同じく温度に近い役割を持つ。データに加える雑音は、重みなしグラフでも構造を変える（ランダム重みの別インスタンスになる）。定温MAで雑音を減らしていくと、そのスケジュール自体が焼きなましに近い役割を持つので、温度だけを変える対照が必要になる。

### 2.3 エネルギーの単調変換と一般化アンサンブル

- **Tsallis–Stariolo [18]、Penna [19]**：受理確率を $[1+(q-1)\Delta E/T]^{-1/(q-1)}$ に一般化する。[18]は提案分布も変えている。
- **STUN [20,21]**：$f=1-\exp(-\gamma(E-E_0))$ を使う。$E_0$ はそれまでの最良値である。$E_0$ の付近は元の地形に近く、高エネルギー側は飽和して障壁が「透明」になる。[21]はγを適応的に決める。
- **Choiの地形変更 [22,23]**：$H^f_{\varepsilon,c}(x)=\int_{H_{\min}}^{H(x)}du/(f((u-c)_+)+\varepsilon)$ を使い、目標を $e^{-H^f_{\varepsilon,c}}$ とする（εが温度）。閾値cより上を平らにし、最小解は保存する。[23]は、イジング型の系で低温でも混合時間を多項式に抑えられる場合があることを示した。
- **Wang–Landau [24]、マルチカノニカル [25]**：状態密度を学習してエネルギーのヒストグラムを平らにする。学習した単調変換とみなせる。
- **含意**
  - 単調変換は局所最小解の集合と順位を変えない。MAでは、実効温度 $T_{\rm eff}(H)=T/\varphi'(H)$ を使うエネルギー依存温度と等価である。例えば $f(u)=u$ のChoi変換では $T_{\rm eff}=\varepsilon+(H-c)_+$ となる。
  - これは時間に依存しない「エネルギーで決まる焼きなまし」である。連鎖は定常なまま、焼きなましに近い効果を持つ（提案D）。
  - STUNのように $E_0$ を最良値で更新すると、履歴依存になる。

### 2.4 履歴依存の地形変更と重み平滑化

- 該当する手法として次のものがある。
  - Energy Landscape Paving [26]、metadynamics [27]
  - タブー探索（GPPへの適用はRollandら [28]）、Guided Local Search [30]、Breakout [29]
  - SAPS [31]：節の重みを平均へ戻す「smoothing」操作を含む
  - PAWS [32]、DLM [33]、max-cutへのBreakout Local Search [34]
- **含意**：訪れた局所最小解を埋めて脱出を促す手法である。GPPではカット辺を特徴量とする重み付けが自然である。ただしMAの定常性が失われ、定温MAとしての解析（温度走査と定常分布の比較）から外れる。本書では主な対象から外し、将来の比較候補にとどめる。

### 2.5 近傍平均、局所自由エネルギー、ベイスン変換

- **連続空間の平滑化**：拡散方程式法 [35] はエネルギーを熱方程式で平滑化し、最小解を追跡する。ガウス型ホモトピーの理論 [37] と段階的非凸化GNC [36] も同じ系統である。
- **局所エントロピー [38–42]**
  - 参照配置xの周りの状態を距離で重み付けした自由エネルギー $F(x)=-\beta^{-1}\log\sum_y\exp(-\beta H(y)-\gamma d(x,y))$ を使う。
  - レプリカを弾性結合したrobust ensembleと、replicated SA [39] がある。量子アニーリングも同じ構造を持つ [42]。
  - 適用先はパーセプトロンとランダムK-SATである。
- **ベイスン変換**：basin hopping [43]、Monte Carlo minimization [44]、large-step Markov chain [45]。探索評価を「局所探索した後の値」に置き換える階段状の地形である。
- **含意**：エネルギーの線形な近傍平均（畳み込み）は、GPPでは温度の付け替えにしかならない（第3章）。非自明な平滑化には、確率（ボルツマン重み）の畳み込み、つまりエネルギーのlog-sum-exp（局所自由エネルギー）が要る。提案Aは、局所エントロピーを半径1に切り詰めて、厳密かつ高速に計算するものと位置づけられる。

### 2.6 地形理論（elementary landscape）

- Grover [46] は、GPPを含む問題で、近傍平均が現在値の一次式になる「波動方程式」を示した。Stadler & Happel [47] と Stadler [48] がこれを理論化した。
- Angel & Zissimopoulos [49] は、罰金係数αつきFlip型二分割地形の自己相関を解析した。
- 関連して、Weinberger [50] の相関長、Whitleyら [51–53] のハミング球上のモーメント計算がある。
- **含意**
  - 目的関数が近傍グラフのラプラシアンの固有関数（と定数の和）なら、近傍平均、球面平均、ランダムウォーク平均はすべて一次変換になる。
  - `gpp`のHは、スピン表示で2次のWalsh項と定数だけからなる。そのため、Flipではalphaによらず固有値4（近傍平均の係数 $1-4/n$）のelementary landscapeである。Swapでもelementaryである（命題1）。
  - [49]はalphaへの依存を論じているが、定義の細部は原典で確認できていない。

### 2.7 確率的評価とpseudo-marginal MCMC

- **pseudo-marginal**：Beaumont [54]、Andrieu & Roberts [55] による。尤度の非負で不偏な推定値を使い、現状態の推定値を保持したまま提案側だけ推定し直すMHは、推定値の期待値に比例する分布を厳密に目標とする。毎回推定し直す方式（MCWM）は近似になる [55,56]。推定の雑音が混合を遅くすることはAndrieu & Vihola [61] の順序づけ結果で知られている（書誌は未照合）。
- **罰金法**：Ceperley & Dewing [57] は、ガウス雑音を含むエネルギー差について、差の分散 $\sigma^2$ を使って受理確率を $\min(1,\exp(-\beta\Delta-\beta^2\sigma^2/2))$ に補正する。
- **ノイズ下のSA** [58–60]。
- **含意**：`gpp`の`random_k_average`は評価値を受理まで保持するので、pseudo-marginal連鎖である（命題3）。その目標分布は、平均評価のボルツマン分布ではなく $\mathbb{E}[e^{-\hat E/T}]$ に比例する分布である。

### 2.8 粗視化、多重レベル、クラスター更新

- 物理系の手法として、Swendsen–Wang [64]、Houdayerのクラスター更新 [65]、Houdayer & Martinの繰り込み [66]、multigrid MC [67] がある。
- 多重レベル分割として、Hendrickson & Leland [68]、METIS [69]、Walshaw [70]、代数的距離 [71]、総説 [72] がある。Walshaw [70] は、多重レベル改良を一般のメタヒューリスティクスを強化する方法として示した。
- **含意**：粗視化は状態空間を縮めて地形を滑らかにし、粗いレベルの局所最小解は少ない。定温MAでも各レベルを同じTで走らせることはできる。ただし提案分布（近傍）の変更になるので、平滑化とは別の軸として扱うのが適切である。

### 2.9 連続緩和、平均場、スペクトル法

- 該当する手法として次のものがある。
  - スペクトル二分割（Fiedler [73]、Pothenら [74]）、拡散型分割 [75]
  - 平均場（Potts）ニューラルネット [76,77]、決定論的アニーリング [78]、graduated assignment [79]
  - 連続二次計画 [80]
- **含意**：`gpp`は連続緩和を扱わない（README）。ただし、スペクトル情報を離散MAの評価関数に取り込むこと（提案B）は、連続緩和には当たらない。

### 2.10 定温MAとGPPの理論・実験の基準

- **Johnsonら [1]**：GPPのSAの実験的評価。罰金形、ランダムグラフ $G_{n,p}$、幾何グラフを使った。
- **定温Metropolis法の理論**
  - Jerrum & Sorkin [81,82]：植え込み二分割モデルで、適切な固定温度のMetropolis法が $O(n^{2+\epsilon})$ ステップで最適二分割を見つける。
  - Carson & Impagliazzo [83]：同じ領域では山登りでも見つかる。
  - Jerrum [84]：Metropolis過程は大きなクリークを見つけられない。
  - Wegener [85]：SAがどの固定温度のMetropolis法にも勝つ、自然な問題（最小全域木の例）を示した。
- **固定温度の実験と解析**
  - Connolly [86]、Cohn & Fielding [87]。
  - Fielding [88]：TSP、QAP、GPPで、最適な固定温度が冷却スケジュールに勝つ場合がある。
  - Orosz & Jacobson [89]：静的SA。
  - Franzin & Stützle [90]：地形の特徴に基づき、近傍構造が一様なら固定温度が有利であることを示した。
- **統計力学**：Fu & Anderson [91]、Banavarら [92]。
- **EO**：Boettcher & Percus [93–96]。EOはランダムグラフでは冪的に収束する。幾何グラフでは結果が明確でないと報告されている（要約情報による）。

### 2.11 まとめと空白

| 系統 | 代表 | 定温MAでの類型 | GPPでの1ステップ費用 | 本書での扱い |
|---|---|---|---|---|
| 近傍平均（線形） | `all_average`、[35–37]の離散版 | 温度等価 | 全近傍平均は $O(n)$ | 命題1で整理 |
| インスタンス平滑化（一律） | [2–5] | 温度とalphaの付け替え | $O(1)$ | 命題2 |
| インスタンス平滑化（構造） | [10–12] | 構造変更 | 行列による | 提案B |
| 確率的近傍評価 | `random_k_average` | 確率的評価（pseudo-marginal） | 実装上 $O(n)$ | 命題3 |
| 局所自由エネルギー | [38–42] | 構造変更 | レプリカ数倍 | 提案A・C |
| 単調変換 | [18–25] | 単調変換 | $O(1)$ | 提案D |
| 履歴依存 | [26–34] | 履歴依存 | $O(\deg)$〜 | 対象外（候補） |
| 粗視化・クラスター | [64–72] | 近傍の変更 | 多様 | 対象外（候補） |

調査の範囲では、次のものは見つからなかった。

1. 重みなしGPPへのGu–Huang型平滑化の適用と、その退化の指摘
2. `random_k`型の標本近傍平均をpseudo-marginal連鎖として解析したもの（一般論 [54,55] は既知）
3. 近傍ソフトミンを評価関数とするMA（提案A）
4. 熱核・スペクトルフィルタによるカットの平滑化ホモトピーを、SAやMAでGPPに使ったもの（提案B。最も近いのは[12]と[75]）
5. replicated SA・局所エントロピーのGPP・max-cutへの適用（提案C）
6. 固定温度でバランス罰金を緩めるホモトピー（4.6節）

いずれも「検索で見つからなかった」ことであり、存在しないことの証明ではない。

## 3. 既存実装の平滑化の理論整理

### 3.1 近傍平均は温度の付け替えにすぎない（命題1）

**命題1** 任意の解sについて次が成り立つ。$C=m/2+\alpha n$ はすべての解にわたるHの平均である。

1. Flipの全近傍平均は $\frac1n\sum_v H(s^{(v)})=(1-4/n)H(s)+2m/n+4\alpha$ である。
2. Flipで半径rのハミング球面の平均は $C+f_r(H(s)-C)$ である。係数は $f_r=\left[\binom{n-2}{r}-2\binom{n-2}{r-1}+\binom{n-2}{r-2}\right]/\binom{n}{r}$。各頂点を独立に確率qで反転した期待値は $C+(1-2q)^2(H(s)-C)$ である。
3. Swapの全近傍平均（等分割上）は $(1-8/n+8/n^2)H(s)+4m/n$ である。Swapの距離2のクラス（2組の同時交換）の平均も、Hの一次式になる。

**証明の要点**：スピン表示では $H=m/2+\alpha n-\frac12\sum_{(i,j)\in E}s_is_j+2\alpha\sum_{i<j}s_is_j$ である。これは2次のWalsh項と定数だけからなる。近傍平均は各2次項に同じ係数を掛けるので、Hの一次式になる。Swapでは、交換による平均カット増分 $4m/n-(8/n-8/n^2)\,\mathrm{cut}$ を直接計算すればよい。Johnsonスキームは距離正則なので、距離2のクラスも同じ固有空間の上で一次式になる。

**系**

- `all_average`を使うMA(T)は、確率法則として平滑化なしのMA $(T/a)$ と同じ連鎖である。係数は、Flipで $a=1-4/n$、Swapで $a=1-8/n+8/n^2$、`weighted_average`（重みw）で $a=1-4w/n$ である（a>0となるのはFlipでn>4、Swapでn≥8のとき）。
  - 個々の軌跡は一致しない。Flipでは、ΔH=0の手の平均評価の差が丸めで±1e-13程度になる。負になると受理判定の乱数を引かないので、乱数列がずれる。受理確率の差は1e-13程度なので、法則には影響しない。
  - 1ステップの費用は、Flipでn倍、Swapで $n^2/4$ 倍になる。得られるのは格子の刻みより小さい温度のずれだけである。
- HCでは、`all_average`で選ぶ手はいつもHの最急降下の手である。ただし、同点の候補の選び方は丸め誤差で決まることがあり、一様でなくなる。ΔH=0の横移動を、Hの局所最適で行うこともある。このため終点が平滑化なしのHCと変わりうる（独立レビューでは、Flipの64回のうち移動列が一致したものはなかった。Swapは4回とも一致した）。
- 温度のずれ $\Delta\Theta=-\log_{10}a$ は次のとおりで、いずれも格子の刻み0.05より小さい。

| n | Flip `all_average` | Swap `all_average` |
|---:|---:|---:|
| 124 | 0.0142 | 0.0287 |
| 250 | 0.0070 | 0.0141 |
| 500 | 0.0035 | 0.0070 |

**数値確認**：n=10、9、8のランダムグラフで全状態を列挙し、上の等式がいずれも $10^{-13}$ 以下の誤差で成り立つことを確かめた（`theory/check_affine.py`、`theory/check_affine2.py`）。

### 3.2 Gu–Huang型平滑化は重みなしGPPで退化する（命題2）

**命題2** 頂点対の重みを辺なら1、辺でなければ0とし、平均を $p=m/\binom n2$ とする。これにGu–Huangの変換を当てると、辺の重みは $p+(1-p)^a$、非辺の重みは $p-p^a$ になる。重み付きカットに罰金を足した評価は、次の形になる。

$$H_a(s)=c_a\,\mathrm{cut}(s)+\Bigl(\alpha-\frac{p-p^a}{4}\Bigr)M(s)^2+\text{const},\qquad c_a=(1-p)^a+p^a$$

したがって、温度Tで $H_a$ を使うMAは、次の条件で平滑化なしのMAを走らせるのと同じである。

- 温度は $T/c_a$ で、元より高い。
- alphaは $(\alpha-(p-p^a)/4)/c_a$ である。

Swap（Mが0で固定）では温度だけの付け替えになる。つまり、重みなしグラフでGu–Huang型のスケジュール（aを大から1へ）を使うと、定温MAに焼きなましと罰金のスケジュールを持ち込むことになる。全列挙で確認した（`theory/check_affine.py`、誤差 $6\times10^{-15}$）。

### 3.3 `random_k_average`はpseudo-marginal連鎖である（命題3）

`gpp`のSAは、候補の平滑化評価を毎回新しく標本化する。一方、現状態の評価は、次に受理されるまで保持する（`src/solvers/engine.rs`の`sa`）。`random_k_average`（Flip、K≤n）の評価は、非復元で一様に選んだK頂点の集合Sについて $\hat E(s;S)=\frac1K\sum_{w\in S}H(s^{(w)})$ である。

**命題3**

1. 連鎖 $(s,S)$ は、拡張空間上の目標 $\bar\pi(s,S)\propto e^{-\hat E(s;S)/T}$ に対するMHである。提案は、sの一様反転とSの独立な引き直しからなる。したがってsの定常分布は次のとおりである。

$$\pi_K(s)\propto\mathbb{E}_S\bigl[\exp(-\hat E(s;S)/T)\bigr]$$

これは平均評価のボルツマン分布 $e^{-\mathbb{E}[\hat E]/T}$ とは異なる（Jensenの不等式）。

2. **K=1**：保持中の反転頂点をwとし、$y=s^{(w)}$ とおく。定常状態では、yは温度Tのボルツマン分布 $e^{-H(y)/T}$ に厳密に従い、wはyと独立に一様に分布する。
   - yの側から見ると、1回の提案は $\{w,v,w'\}$ の対称差を反転することになる（vは提案頂点、w'は新しい標本）。提案はほとんど3頂点の反転であり、1頂点の反転になる確率はおよそ3/nにすぎない。
   - したがってK=1は目標分布を平滑化しない。変わるのは提案の形（大きく、低温では受理されにくい）と、観測に1頂点分のずれが乗ることだけである。
3. **K≥2**
   - 仮に復元抽出なら、有効エネルギーは $F_K(s)=-KT\log\frac1n\sum_w e^{-H(s^{(w)})/(KT)}$ であり、軟らかさKTの近傍ソフトミンになる。
   - `gpp`は非復元抽出なので、厳密な法則は1.の $\pi_K$ である。復元抽出の $F_K$ との差は低温で大きい。独立レビューでは、n=124・K=2で状態ごとの差 $(F_{\rm 復元}-F_{\rm 非復元})/T$ の標準偏差が、Θ=−1.5で2.1、−1で0.55、−0.5で0.15、0で0.007だった。$F_K$ による解釈はΘ≳−0.5でだけ使える。
   - 高温側（KTが利得の広がりより十分大きいとき）では、非復元抽出のキュムラント展開を2次で打ち切って $F_K(s)\approx(1-4/n)H(s)+\text{const}-\frac{n-K}{n-1}\cdot\frac{\sigma_g^2(s)}{2KT}$ となる。$\sigma_g^2(s)$ はFlip利得 $g_v(s)$ の分散である。係数 $(n-K)/(n-1)$ は正しく、誤差は $1/T^2$ で小さくなる。ただし低温では近似が悪く、補正しない場合より悪くなることもある。
   - 元の地形からのずれは、強く改善する手を持つ状態を優遇する項であり、Kについて1/Kで弱まる。K=nでは全近傍平均（命題1）に一致する。
   - 拡張空間で見ると、K≥2は「中心sの1頂点反転にあたるK個のレプリカを、それぞれ温度KTで持つ系」である。局所エントロピーのrobust ensemble [39] を、硬い拘束で書いたものにあたる。

**数値確認**：n=8、|E|=10のグラフで、`engine.rs`と同じ手順の連鎖を $2\times10^6$ ステップ回した。経験分布との全変動距離（TV）は次のとおりである（`theory/check_pseudo_marginal.py`）。

| T | K | 理論 $\pi_K$ | $e^{-\text{平均}/T}$ | $e^{-H/T}$ | K=1の影の解yと $e^{-H/T}$ |
|---:|---:|---:|---:|---:|---:|
| 0.7 | 1 | **0.012** | 0.160 | 0.359 | **0.008** |
| 0.7 | 2 | **0.009** | 0.086 | 0.303 | |
| 0.7 | 4 | **0.008** | 0.033 | 0.271 | |
| 1.5 | 1 | **0.008** | 0.045 | 0.176 | **0.006** |
| 1.5 | 2 | **0.007** | 0.022 | 0.166 | |
| 1.5 | 4 | **0.007** | 0.010 | 0.160 | |

理論値とのTVは、状態数256に対する標本誤差の水準（0.01前後）にある。平均評価のボルツマン分布とのずれは、Kが小さく温度が低いほど大きい。

独立レビューでは次の2つの方法でも確かめた。どちらもFlipとSwapの両方で成り立った。

- 拡張連鎖（sと保持中の標本）の遷移行列を厳密に解いた。sの周辺分布は $\pi_K$ に一致した（TV≤1e-15）。K=1の影の解はボルツマン分布に一致した（TV≤3e-16）。
- `gpp`の`smoothing::evaluate`をそのまま使い、$8\times10^8$ ステップまでシミュレーションした。$\pi_K$ とのTVは0.0004〜0.0051だった。

Swapでは、Sは $(n/2)^2$ 個の交換から選ぶK個の部分集合である。Kがこれを超えると距離2の状態も標本に入るので、$\pi_K$ にもそれを含める必要がある。

**実装上の注記**：`smoothing::evaluate`は評価のたびに距離1の候補の一覧を作る。そのため`random_k_average`の1ステップはK=1でもFlipで $O(n)$、Swapで $O(n^2/4)$ かかる。計算結果には影響しないが、Swapで使う場合は所要時間に効く。

### 3.4 復旧run Bへの含意（予測）

B（Flip×SA、Θ81点、none・`random_k` K=1〜32・`all_average`）の結果は、次のように予測できる。

- **`all_average`**：noneの曲線をΘ方向に0.014（n=124）、0.007（n=250）、0.0035（n=500）だけずらしたものに一致する（命題1）。Bのこの系列は、復旧計画の意図どおり、この理論の確認になる。
- **K=1**：影の解は温度Tのボルツマン分布のままである。提案が3頂点の反転に近いため、低温で受理率が大きく下がる。最適なΘは高温側へずれ、最適点での値はnoneと同等か悪くなると予想する。予備実験の結果は第5章を参照。
- **K≥2**：Θ≳−0.5では、軟らかさKTのソフトミンとして解釈できる。それより低温では、現行解が良い解の隣に留まる傾向が強まり、受理率が大きく下がる。Kを大きくすると、全近傍平均つまりnoneへ近づく。小さいKで効果がありうるが、Kについて単調とは限らない。
- **計測上の注意**：Kが小さいと、現行解の実評価値に1頂点分のずれが乗る。現行解や暫定解の生の値ではなく、ベイスン値で比べるべきである。

## 4. 新規手法の提案

### 4.1 設計の指針

第3章から、定温MAで意味のある平滑化は次の条件を満たす必要がある。

1. 温度と等価でない（Θの走査で置き換えられない）。
2. 局所最小解の集合か、障壁の構造を変える。
3. 1ステップの計算量が平滑化なしと同程度（$O(\deg)$）である。
4. 決定的で、乱数の消費を増やさない（`gpp`の再現性の規則に合う）。
5. 連続なパラメータを持ち、その極限で元の問題に戻る（静的にもホモトピーにも使える）。
6. ベイスン計測と整合する。

### 4.2 提案A：近傍ソフトミン平滑化（NSM）

**定義** 軟らかさλ>0と近傍重みκ≥0に対し、次の評価を使う。

$$F_{\lambda,\kappa}(s)=-\lambda\log\Bigl(e^{-H(s)/\lambda}+\kappa\sum_{y\in N(s)}e^{-H(y)/\lambda}\Bigr)=H(s)-\lambda\log\Bigl(1+\kappa\sum_v e^{-g_v(s)/\lambda}\Bigr)$$

実装ではλを温度に比例させ、$\lambda=\rho T$ とする。

**命題4**（性質）

1. $F_{\lambda,\kappa}\le H$ である。κ=0ならHに一致する。κは元の問題へ戻すホモトピーのパラメータになる。
2. λ→0（κ>0）で、エロージョン $F_0(s)=\min_{y\in N[s]}H(y)$（閉近傍での最小値）に収束する。
3. λ→∞では、$(H+\kappa\sum_yH(y))/(1+n\kappa)$ に定数を足した形に近づく。命題1によりこれはHの一次式であり、温度等価な平滑化に戻る。
4. λ=Tでは $e^{-F/T}=e^{-H(s)/T}+\kappa\sum_{y\in N(s)}e^{-H(y)/T}$ となる（近傍グラフが正則、つまり全状態で近傍数が等しいことを使う。FlipとSwapは満たす）。目標分布は、ボルツマン分布を核（中心1、近傍κ）で畳み込んだものである。$N[s]$ から事後分布 $\propto k(s,y)e^{-H(y)/T}$ に従ってyを引けば、yは厳密に温度Tのボルツマン標本になる。つまり、平衡での質を保ったまま、地形だけを滑らかにした連鎖である。
5. `random_k_average`の目標分布は、復元抽出の場合、λ=KTとし、中心の項 $e^{-H(s)/\lambda}$ を除いたNSMの目標分布に一致する（命題3。κは定数のずれにしかならない）。`gpp`の非復元抽出では、高温側（Θ≳−0.5）で近似的に一致する。K=1なら厳密に一致する。`random_k_average`はこれをpseudo-marginalで雑音つきに実行している。NSMはこれを決定的にし、λをTから切り離し、κで元の問題へ戻せるようにした一般化である。一般に、推定の雑音は混合を悪くする方向に働く [61]。
6. エロージョン地形 $F_0$ では次が成り立つ。
   1. $F_0$ の局所最小解sで閉近傍の最良解 $y^*(s)$ をとると、$y^*(s)$ はHの局所最小解である。
   2. Hの局所最小解 $y_1,y_2$ のハミング距離が3以下なら、両者を結ぶ経路で $F_0$ が $\max(H(y_1),H(y_2))$ を超えないものがある。つまり、ハミング距離3以内の局所最小解の間の障壁は消える。
   - GPPのFlipでは、Swap1回（距離2）や3頂点の移動がこれに当たる。Flip-MAが、罰金とカットの障壁を越えずにSwapを実行できるようになる。

**証明の要点（6）**

- 1.について：$y^*$ がsと等しい場合と、sの1頂点反転である場合に分ける。どちらの場合も、$y^*$ がより低い近傍を持つなら、sの近傍に $F_0$ がより小さい点ができて矛盾する。
- 2.について：距離3なら $y_2=y_1\oplus\{a,b,c\}$ とし、経路 $y_1\to y_1^{(a)}\to y_1^{(a,b)}\to y_2$ をとる。中間の2点はそれぞれ $y_1$、$y_2$ の近傍なので、$F_0\le H(y_1)$、$F_0\le H(y_2)$ となる。
- 全列挙での確認（その1）：n=12のグラフ12個（幾何・ランダム各6）で1.と2.を調べた（`theory/check_erosion.py`）。1.の反例はなかった。距離2・3の局所最小解の組は、すべて障壁なしで結べた。距離4では結べない組があった（例：10組中0組）。
- 全列挙での確認（その2）：n=16のFlip＋罰金の地形で調べた（`theory/landscape2.py`）。台地を含む局所最小解の数は、幾何グラフで10.8→2.5（エロージョン）、4.8（λ=0.5, κ=1）となった。ランダムグラフでは30.0→4.2、7.8となった。全初期解からの最急降下が最適解に至る割合は、幾何グラフで0.56→0.78、ランダムグラフで0.46→0.71（いずれもλ=0.5, κ=1）に増えた。

**計算法**：Flip利得のカット部分 $d_v-2c_v$ は $[-D,D]$ の整数である（Dは最大次数）。これを群ごとの整数バケット（FM法 [63] やn-fold way [62] と同じ構造）で数える。

- 罰金部分は群ごとの定数 $4\alpha(1\mp M)$ なので、$\sum_v e^{-g_v/\lambda}$ は2群×(2D+1) 個のバケットの和で求まる。
- 最小利得でずらした $\log(1+e^x)$ で計算すれば、あふれは起きない。
- 1ステップは次の手順になる。
  1. 候補を仮に適用する。v自身と隣接頂点のバケットを更新する（$O(\deg v)$）。
  2. Fを評価する（$O(D)$）。
  3. 棄却したら元に戻す。
- 乱数は平滑化なしと同じく、選択に1個、受理判定に1個だけ使う。
- Swap版では、交換の利得 $g_a+g_b+2A_{ab}$（$g$ はカット部分の利得。等分割なので罰金は0）について、$\sum e^{-\cdot/\lambda}=Q_AQ_B+\sum_{\text{カット辺}}e^{-(g_a+g_b)/\lambda}(e^{-2/\lambda}-1)$ を使う。$Q_A=\sum_{a\in A}e^{-g_a/\lambda}$ は群ごとのバケットから求まる。補正項は、カット辺を $g_a+g_b$ の値ごとに数える整数ヒストグラムを持てばよい。評価は $O(D)$、更新は $O(\deg^2)$ で済み、浮動小数点の累積誤差も生じない。この式は、n=12の全等分割で直接計算と一致することを確かめた（`theory/check_swap_softmin.py`）。

**パラメータと仮説**

- 静的な使い方：$\rho\in\{1,1/2,1/4,0\}$（0はエロージョン）、κ=1。
- ホモトピー：κを1から0へ下げる。
- 仮説A1：ρ=1では平衡での質を保ったまま混合が速くなる。
- 仮説A2：ρ<1では、Swap相当の障壁が消えることで、幾何グラフで改善する。
- 仮説A3：同じλ=KTの`random_k`より分散が小さく、1ステップも速い。

**新規性**：局所エントロピー [38–42] を半径1に切り詰め、バケットで厳密に評価してMAの評価関数に使う例は、調査の範囲では見つからなかった。

### 4.3 提案B：熱核スペクトル平滑化（HKS）

**動機**：命題2から、重みなしグラフで意味のあるインスタンス平滑化には、一律でない重みの変更が要る。Gu–Huangの考え方は「易しいインスタンスから本物へ連続につなぐ」ことである。その経路を、グラフの構造を保つ拡散で作る。

**定義** ラプラシアンをLとし、$W(t)=\mathrm{offdiag}(e^{-tL})$ を、辺重みの総和が $2m$ になるように $c_t$ 倍して正規化する。

$$H_t(s)=\sum_{i<j,\ s_i\ne s_j}W_{ij}(t)+\alpha M(s)^2=\frac{c_t}{4}\sum_{k\ge2}(1-e^{-t\lambda_k})(u_k\cdot s)^2+\alpha M(s)^2$$

$(\lambda_k,u_k)$ はLの固有対である。$e^{-tL}\mathbf 1=\mathbf 1$ なので、Wのラプラシアンは $c_t(I-e^{-tL})$ に等しい。時間は $t=\tau/\bar d$ と平均ホップ数τで表す。

**命題5**（2.〜4.は連結なグラフの場合）

1. t→0で $c_t\approx1/t$ となり、$H_t\to H$ である。
2. t→∞では、等分割上で $H_t$ は定数（一様完全グラフのカット）に近づく。これはGu–Huangのa→∞にあたる自明な問題である。非連結なグラフでは、各連結成分の中で一様な重みに近づき、成分をどちらの群へ置くかの問題が残る。
3. 残る主な起伏は $-\frac{c_t}{4}e^{-t\lambda_2}(u_2\cdot s)^2$ である。その最小化は、フィードラーベクトルの中央値で分けるスペクトル二分割 [73,74] になる。この向きだけを見るとSwap近傍で単峰である。
4. 一様極限では、カットの部分が $-\bar w M^2/4$（$\bar w=\bar d/(n-1)$）の項を含む。つまり不均衡をわずかに好む。ベースラインのグラフでは $\bar d/(4(n-1))\le0.042<\alpha$ なので、罰金は効いたままである。

ベースラインの18グラフのうち、平均次数5の6グラフとgeometric n=250・次数10は非連結である（連結成分は3〜20個）。

**性質と費用**

- 幾何グラフでは、熱核は幅 $\sim\sqrt\tau$ ×接続半径のガウス型の空間フィルタとして働く。平滑化した問題は、境界の長さを最小にする連続問題に近づき、格子規模のでこぼこ（ドメイン壁のピン止め）を抑えると期待できる。
- ランダムグラフはエキスパンダーに近く（$\lambda_2$ が大きい）、スペクトルの情報が弱い。効果は小さいと予想する。
- 計算量は次のとおりである。
  - 前処理：固有分解 $O(n^3)$。
  - 1ステップ：局所場 $h_i=\sum_jW_{ij}s_j$ を保持すれば、評価は $O(1)$、受理時の更新は $O(n)$。
  - n≤500では密行列（2 MB）で足りる。大規模では多項式近似でkホップに打ち切る（$O(\bar d^k)$）。
  - 乱数は使わない。

**温度との分離**

- 静的に使う場合、正規化による全体の尺度はΘの走査で吸収される。
- τを時間とともに下げる場合、正規化は実効温度のスケジュールを伴う。大きなτでは起伏の振幅が $e^{-t\lambda_2}$ で縮み、高温と同じ効果を持つ。この場合は、同じ実効温度の推移を与える平滑化なしの焼きなましを対照にする。

**パラメータと仮説**

- 静的：$\tau\in\{0.25,0.5,1,2\}$。スケジュール：τを4から半減させて0へ（最後の20%は元の問題）。
- 仮説B1：幾何グラフで改善し、ランダムグラフではほぼ中立。
- 仮説B2：スケジュールの効果は、温度だけを同じように動かした対照を上回る。

**新規性**：熱核やスペクトルフィルタでカットそのものを平滑化し、SA・MAのホモトピーとしてGPPに使った例は、調査の範囲では見つからなかった。最も近いのは、スピングラスの結合行列のべき乗変換 [12] と拡散型分割 [75] である。

### 4.4 提案C：ラベル対称なレプリカ結合MA（局所エントロピーのGPP版）

**定義** y個のレプリカ $s^1,\dots,s^y$ を温度Tで動かし、次の評価を使う。

$$E(s^1,\dots,s^y)=\sum_aH(s^a)-\frac{\gamma}{n}\sum_{a<b}(q_{ab})^2,\qquad q_{ab}=s^a\cdot s^b$$

$(q_{ab})^2$ は各レプリカの群ラベルの入れ替え $s^a\to-s^a$ で変わらない。GPPの解は群ラベルの入れ替えで同じ分割を表すので、この対称性を持たない結合（ハミング距離）は同じ分割を最も遠いものと扱ってしまう。

**計算法**：レプリカaの頂点vを反転したときの差分は $g_v(s^a)-\frac{\gamma}{n}\sum_{b\ne a}(4-4s^a_vs^b_vq_{ab})$ である。重なり $q_{ab}$ を保持すれば、1ステップ $O(\deg+y)$ で計算できる。

**運用**

- 予算：全レプリカの提案数の合計を $10^6$ とし、同じ計算量で比べる。
- γのスケジュール：0から $\gamma_{\max}\approx\bar d/(4(y-1))$ へ上げる。γが焼きなましの役を担う。静的なγも試す。
- 読み出し：全レプリカの実評価の最良値とベイスン値。

**位置づけ**：命題3のとおり、`random_k`（K≥2）は、この系を「中心から1頂点反転の範囲」という硬い拘束で書いたものにあたる。

**仮説と新規性**：広い（エントロピーの大きい）局所最小解と深い解が相関すれば改善すると予想するが、GPPでこの相関があるかは未知である。GPP・max-cutへの適用例と、ラベル対称性の扱いを論じた例は、調査の範囲では見つからなかった。

### 4.5 提案D：データ駆動のエネルギー依存温度（対照を兼ねる）

**定義** 単調変換 $\varphi$ を $\varphi'(E)=T/T^*(E)$ と決め、温度Tで $\varphi(H)$ を使うMAを走らせる。$T^*(E)$ は、ベースラインAの81温度・56計測点の軌跡から推定する。具体的には、エネルギー帯Eで改善の速度（$\log t$ あたりの暫定解ベイスンの減少）が最大になる温度とする。

**性質**

- この連鎖は定常で、状態がエネルギーEにあるときは温度 $T^*(E)$ のMAとして振る舞う。時間に依存しない焼きなましとみなせる。
- 局所最小解は変えないので、構造変更型の平滑化（A〜C）とは効き方が異なる。
- 比較での役割：A〜Cの改善が「エネルギーに応じて温度を変えるだけ」で得られるものかを判定する対照になる。
- 先行手法 [18–23] の、データに基づく具体化にあたる。

### 4.6 その他の候補（優先度は低い）

- **バランス罰金の緩和**
  - 内容：Flipで、不感帯つき罰金 $\alpha\max(0,|M|-b)^2$ のbを縮めていく、またはalphaを段階的に上げる。
  - 評価：alpha=0.05では、M≈0での1回の反転の罰金は $4\alpha(1\mp M)\approx0.2$ にすぎない。典型的なカット利得（次数程度）より十分小さいので、障壁の構造への影響は小さく、主にMの分布を変えると予想する。
  - 新規性：固定温度でこれを行う先行例は見つからなかったが、単純なので対照として使う程度にとどめる。
- **データへのノイズ（ノイズ法 [13–15]）**：辺重みに雑音を加え、雑音を減らしていく。温度だけを変える対照が必須である。
- **粗視化（多重レベル）MA、ベイスン変換（ILS型のMA）**：近傍の変更として別の軸で扱う。

### 4.7 優先順位

| 提案 | 期待 | 実装の手間 | 優先度 |
|---|---|---|---|
| A NSM | `random_k`を置き換え、Swap相当の障壁を除く | バケットの実装（EOの索引に近い） | 高 |
| B HKS | 幾何グラフでドメイン壁を扱う | 固有分解と密な局所場 | 高（幾何グラフ） |
| D エネルギー依存温度 | 焼きなまし効果の上限と対照 | $O(1)$、ベースラインAの結果が必要 | 中 |
| C レプリカ結合 | 広い解への誘導 | 重なりの保持、予算の定義 | 中 |
| バランス緩和・ノイズ法 | 対照 | 小 | 低 |

## 5. 予備実験（ベースライン外のプロトタイプ）

試走を実行中である。結果は集計後にこの章へ追記する。

## 6. 実験計画と実装方針

### 6.1 条件

- グラフ、ステップ数、実行シード、alpha、計測はベースラインのとおりとする。
  - グラフ：18組×s0
  - ステップ数：$10^6$
  - 実行シード：0〜31
  - 計測：対数間隔56点。現行解と暫定解のベイスン値を測る（`max_basin_steps=10000`）
- Θ格子はベースラインの81点とする。none列はAの結果を再利用する。
- 新しい軸は平滑化パラメータだけにする。

### 6.2 系列と規模

| 系列 | 値 | ジョブ数 |
|---|---|---:|
| NSM | ρ∈{1, 1/2, 1/4, 0}、κ=1（Flip） | 186,624 |
| HKS静的 | τ∈{0.25, 0.5, 1, 2}（Flip） | 186,624 |
| HKSスケジュール | τ：4から半減して0へ（Flip） | 46,656 |
| 焼きなましの対照 | HKSスケジュールと同じ実効温度の推移 | 46,656 |

ジョブ数は、系列数×81温度×18グラフ×32シードである。ベースラインAの見積もり（163,584ジョブで113 CPU時間）から、1ジョブ約2.5 CPU秒（大半がベイスン計測）とすると、NSMとHKS静的はそれぞれ約130 CPU時間（12並列で約11時間）になる。まずシード0〜7でふるい分け、有望な系列だけ0〜31へ広げる。条件IDはシードを含まないので、後で足したシードの結果もそのまま使える。

### 6.3 指標

- 主指標：各グラフで、最適Θにおける $10^6$ ステップ時点の暫定解ベイスン値の平均。
  - Θの選択による楽観的な偏り（勝者の呪い）を避けるため、シードを2分割して交差検証で選ぶ。
- 副指標
  - 目標値到達時間：既知の最良値からx%以内に入る最初の計測点
  - Θに対する頑健さ：最適値から1%以内に入るΘの幅
  - 1ステップあたりのCPU時間
  - 受理率

### 6.4 対照

- 平滑化なし（Aの結果）。
- スケジュール系列には、同じ実効温度の推移を与える焼きなまし（提案D、または受理率を合わせたスケジュール）。
- 命題1の確認として、Bの`all_average`。

### 6.5 `gpp`での実装方針

[extending.md](extending.md)に従う。

- 設定：`SmoothingSpec`に`soft_min { lambda_ratio, kappa }`と`heat_kernel { tau }`を追加する。スケジュールは別途定義する。既存の種類のシリアライズとIDは変えない。
- 状態：平滑化の状態オブジェクト（NSMのバケット、HKSの密行列と局所場）をジョブが持ち、差分で更新する。EOの差分更新索引と同じ考え方である。
- 乱数：追加の消費はない（NSMもHKSも決定的）。
- 浮動小数点：バケットの和は添字の順に取り、演算順を固定する。
- 検証：`exact_tests`に、全近傍を列挙してFを直接計算する独立参照を置き、小規模グラフで各ステップのビット一致を確かめる。バケットの整数カウントが全再計算と一致することも確かめる。
- 適用範囲：HCでNSMを使うと先読み山登りになる。EOは平滑化を禁止する現行の規則のままとする。
- 版：新しい種類は新しい条件IDになる。既存条件の`versions.algorithm`は据え置く。

### 6.6 Bの結果で確かめること

1. `all_average`：noneをΘ方向に命題1の量だけずらしたものと、統計誤差の範囲で一致するか。
2. K=1：受理率がnoneより大きく下がり、最適Θが高温側へずれるか（命題3）。
3. K≥2：Kを大きくするとnoneに近づくか。

## 7. 限界と未解決事項

- **検証の体制**：命題1〜4と予備実験のコードは、実装した者とは別のエージェントが独立に確かめた（[AGENTS.md](../AGENTS.md)の方針）。命題1・2は有理数での厳密計算、命題3は遷移行列の厳密解と`gpp`本体を使ったシミュレーション、NSMとHKSの実装は全近傍の直接計算・`scipy`の行列指数関数との照合で確かめ、結果を変えるバグは見つからなかった。指摘を受けて、命題1の系（HCと浮動小数点）と命題3の近似の適用範囲を本文に追記した。
- **文献の確認**：Web検索の要約情報に頼っている。原典で数値や定義を確かめていない項目がある（第8章に明記）。検索予算を使い切ったため、[8]の著者名などは確認できなかった。
- **予備実験**：`gpp`の外の自作実装で、シード8個・Θ23点・最終点のベイスンだけを測った。ベースラインの計測（56点、同点処理の乱数）とは異なる。
- **理論の範囲**：命題3は定常分布についての主張である。$10^6$ ステップという有限時間の挙動は、別に扱う必要がある。
- **Swap版**：NSMのSwap版は和の式を全列挙で確かめただけで、探索としては試していない。HKSのSwap版も試していない。

## 8. 参考文献

書誌は調査時にWeb検索の結果で確認したものである。「未照合」と記したものは、細部（巻号・頁・著者）を確認できていない。

**基準・GPP・EO**

1. D. S. Johnson, C. R. Aragon, L. A. McGeoch, C. Schevon, "Optimization by simulated annealing: An experimental evaluation; Part I, graph partitioning," *Operations Research* 37(6):865–892, 1989. doi:10.1287/opre.37.6.865

**探索空間平滑化**

2. J. Gu, X. Huang, "Efficient local search with search space smoothing: A case study of the traveling salesman problem (TSP)," *IEEE Trans. Systems, Man, and Cybernetics* 24(5):728–735, 1994.
3. J. Schneider, M. Dankesreiter, W. Fettes, I. Morgenstern, M. Schmid, J. M. Singer, "Search-space smoothing for combinatorial optimization problems," *Physica A* 243(1–2):77–112, 1997. doi:10.1016/S0378-4371(97)00207-0
4. S. P. Coy, B. L. Golden, G. C. Runger, E. A. Wasil, "See the forest before the trees: Fine-tuned learning and its application to the traveling salesman problem," *IEEE Trans. SMC-A* 28(4):454–464, 1998.
5. S. P. Coy, B. L. Golden, E. A. Wasil, "A computational study of smoothing heuristics for the traveling salesman problem," *European J. Operational Research* 124(1):15–27, 2000. doi:10.1016/S0377-2217(99)00125-3
6. J. Schneider, S. Kirkpatrick, *Stochastic Optimization*, Springer, 2006. doi:10.1007/978-3-540-34560-2（章の内容は未照合）
7. 久保幹雄, J. P. ペドロソ, 『メタヒューリスティクスの数理』, 共立出版, 2009（2.8節「探索空間平滑化法・交互平滑化法」）.
8. 「メトロポリスアルゴリズムとの併用による探索空間平滑化法の機能特性 I・II」, 日本機械学会 最適化シンポジウム2012 講演論文集, 講演番号2208・2209（J-STAGE: https://www.jstage.jst.go.jp/article/jsmeopt/2012.10/0/2012.10__2209-1_/_article/-char/ja/ 。著者名は未照合）.
9. J. Sun ほか, "Homotopic convex transformation: A new landscape smoothing method for the traveling salesman problem," *IEEE Trans. Cybernetics*, doi:10.1109/TCYB.2020.2981385（巻号未照合）.
10. "A new parallel cooperative landscape smoothing algorithm and its applications on TSP and UBQP," arXiv:2401.03237, 2024（著者表記・掲載誌は未照合）.
11. "On the effects of smoothing rugged landscape by different toy problems: A case study on UBQP," IEEE CEC 2024, arXiv:2407.19676（著者表記は未照合）.
12. Ya. M. Karandashev, B. V. Kryzhanovsky, "Matrix-power energy-landscape transformation for finding NP-hard spin-glass ground states," *J. Global Optimization*, 2015, doi:10.1007/s10898-014-0153-7（巻号未照合）.

**ノイズ法・データ摂動**

13. I. Charon, O. Hudry, "The noising method: A new method for combinatorial optimization," *Operations Research Letters* 14(3):133–137, 1993. doi:10.1016/0167-6377(93)90023-A
14. I. Charon, O. Hudry, "The noising methods: A generalization of some metaheuristics," *European J. Operational Research* 135(1):86–101, 2001. doi:10.1016/S0377-2217(00)00305-2
15. V. Sudhakar, C. Siva Ram Murthy, "A modified noising algorithm for the graph partitioning problem," *Integration, the VLSI Journal* 22:101–113, 1997.
16. R. H. Storer, S. D. Wu, R. Vaccari, "New search spaces for sequencing problems with application to job shop scheduling," *Management Science* 38(10):1495–1509, 1992.
17. B. Codenotti, G. Manzini, L. Margara, G. Resta, "Perturbation: An efficient technique for the solution of very large instances of the Euclidean TSP," *INFORMS J. Computing* 8(2):125–133, 1996.

**単調変換・一般化アンサンブル**

18. C. Tsallis, D. A. Stariolo, "Generalized simulated annealing," *Physica A* 233:395–406, 1996.
19. T. J. P. Penna, "Traveling salesman problem and Tsallis statistics," *Physical Review E* 51(1):R1–R3, 1995.
20. W. Wenzel, K. Hamacher, "Stochastic tunneling approach for global minimization of complex potential energy landscapes," *Physical Review Letters* 82(15):3003–3007, 1999.
21. K. Hamacher, "Adaptation in stochastic tunneling global optimization of complex potential energy landscapes," *Europhysics Letters* 74(6):944–950, 2006.
22. M. C. H. Choi, "Improved Metropolis–Hastings algorithms via landscape modification with applications to simulated annealing and the Curie–Weiss model," arXiv:2011.09680（*Advances in Applied Probability*掲載、巻号未照合）.
23. M. C. H. Choi, "Landscape modification meets spin systems: from torpid to rapid mixing, tunneling and annealing in the low-temperature regime," arXiv:2208.10054.
24. F. Wang, D. P. Landau, "Efficient, multiple-range random walk algorithm to calculate the density of states," *Physical Review Letters* 86(10):2050–2053, 2001.
25. B. A. Berg, T. Neuhaus, "Multicanonical ensemble: A new approach to simulate first-order phase transitions," *Physical Review Letters* 68:9–12, 1992.

**履歴依存・重み平滑化**

26. U. H. E. Hansmann, L. T. Wille, "Global optimization by energy landscape paving," *Physical Review Letters* 88:068105, 2002.
27. A. Laio, M. Parrinello, "Escaping free-energy minima," *PNAS* 99(20):12562–12566, 2002.
28. E. Rolland, H. Pirkul, F. Glover, "Tabu search for graph partitioning," *Annals of Operations Research* 63:209–232, 1996.
29. P. Morris, "The breakout method for escaping from local minima," *Proc. AAAI-93*, pp. 40–45, 1993.
30. C. Voudouris, E. Tsang, "Guided local search and its application to the traveling salesman problem," *European J. Operational Research* 113(2):469–499, 1999（巻号未照合）.
31. F. Hutter, D. A. D. Tompkins, H. H. Hoos, "Scaling and probabilistic smoothing: Efficient dynamic local search for SAT," *CP 2002*, LNCS 2470, 2002.
32. J. Thornton, D. N. Pham, S. Bain, V. Ferreira Jr., "Additive versus multiplicative clause weighting for SAT," *Proc. AAAI-04*, pp. 191–196, 2004.
33. Y. Shang, B. W. Wah, "A discrete Lagrangian-based global-search method for solving satisfiability problems," *J. Global Optimization*, 1998, doi:10.1023/A:1008287028851.
34. U. Benlic, J.-K. Hao, "Breakout local search for the max-cut problem," *Engineering Applications of Artificial Intelligence* 26(3):1162–1173, 2013.

**平滑化の連続版・局所エントロピー・ベイスン変換**

35. L. Piela, J. Kostrowicki, H. A. Scheraga, "The multiple-minima problem in the conformational analysis of molecules. Deformation of the potential energy hypersurface by the diffusion equation method," *J. Physical Chemistry* 93:3339–3346, 1989.
36. A. Blake, A. Zisserman, *Visual Reconstruction*, MIT Press, 1987.
37. H. Mobahi, J. W. Fisher III, "A theoretical analysis of optimization by Gaussian continuation," *AAAI 2015*, pp. 1205–1211.
38. C. Baldassi, A. Ingrosso, C. Lucibello, L. Saglietti, R. Zecchina, "Subdominant dense clusters allow for simple learning and high computational performance in neural networks with discrete synapses," *Physical Review Letters* 115:128101, 2015.
39. C. Baldassi, C. Borgs, J. T. Chayes, A. Ingrosso, C. Lucibello, L. Saglietti, R. Zecchina, "Unreasonable effectiveness of learning neural networks: From accessible states and robust ensembles to basic algorithmic schemes," *PNAS* 113(48):E7655–E7662, 2016.
40. C. Baldassi, A. Ingrosso, C. Lucibello, L. Saglietti, R. Zecchina, "Local entropy as a measure for sampling solutions in constraint satisfaction problems," *J. Statistical Mechanics* 2016:023301.
41. P. Chaudhari ほか, "Entropy-SGD: Biasing gradient descent into wide valleys," *ICLR 2017*; *J. Statistical Mechanics* 2019:124018.
42. C. Baldassi, R. Zecchina, "Efficiency of quantum vs. classical annealing in nonconvex learning problems," *PNAS* 115(7):1457–1462, 2018.
43. D. J. Wales, J. P. K. Doye, "Global optimization by basin-hopping and the lowest energy structures of Lennard-Jones clusters containing up to 110 atoms," *J. Physical Chemistry A* 101(28):5111–5116, 1997.
44. Z. Li, H. A. Scheraga, "Monte Carlo-minimization approach to the multiple-minima problem in protein folding," *PNAS* 84(19):6611–6615, 1987.
45. O. Martin, S. W. Otto, E. W. Felten, "Large-step Markov chains for the traveling salesman problem," *Complex Systems* 5(3):299–326, 1991.

**地形理論**

46. L. K. Grover, "Local search and the local structure of NP-complete problems," *Operations Research Letters* 12(4):235–243, 1992. doi:10.1016/0167-6377(92)90049-9
47. P. F. Stadler, R. Happel, "Correlation structure of the landscape of the graph-bipartitioning problem," *J. Physics A* 25:3103–3110, 1992.
48. P. F. Stadler, "Landscapes and their correlation functions," *J. Mathematical Chemistry* 20(1):1–45, 1996. doi:10.1007/BF01165154
49. E. Angel, V. Zissimopoulos, "Autocorrelation coefficient for the graph bipartitioning problem," *Theoretical Computer Science* 191:229–243, 1998（題名の表記揺れあり、内容は未照合）.
50. E. D. Weinberger, "Correlated and uncorrelated fitness landscapes and how to tell the difference," *Biological Cybernetics* 63:325–336, 1990.
51. L. D. Whitley, A. M. Sutton, A. E. Howe, "Understanding elementary landscapes," *GECCO 2008*.
52. A. M. Sutton, L. D. Whitley, A. E. Howe, "Computing the moments of k-bounded pseudo-Boolean functions over Hamming spheres of arbitrary radius in polynomial time," *Theoretical Computer Science* 425:58–74, 2012.
53. F. Chicano, L. D. Whitley, A. M. Sutton, "Efficient identification of improving moves in a ball for pseudo-Boolean problems," *GECCO 2014*, pp. 437–444.

**確率的評価・pseudo-marginal**

54. M. A. Beaumont, "Estimation of population growth or decline in genetically monitored populations," *Genetics* 164:1139–1160, 2003（題名は未照合）.
55. C. Andrieu, G. O. Roberts, "The pseudo-marginal approach for efficient Monte Carlo computations," *Annals of Statistics* 37(2):697–725, 2009. doi:10.1214/07-AOS574
56. F. J. Medina-Aguayo, A. Lee, G. O. Roberts, "Stability of noisy Metropolis–Hastings," *Statistics and Computing*, 2016, doi:10.1007/s11222-015-9604-3.
57. D. M. Ceperley, M. Dewing, "The penalty method for random walks with uncertain energies," *J. Chemical Physics* 110(20):9812–9820, 1999.
58. S. B. Gelfand, S. K. Mitter, "Simulated annealing with noisy or imprecise energy measurements," *J. Optimization Theory and Applications*, 1989, doi:10.1007/BF00939629.
59. W. J. Gutjahr, G. Ch. Pflug, "Simulated annealing for noisy cost functions," *J. Global Optimization* 8:1–13, 1996.
60. J. Branke, S. Meisel, C. Schmidt, "Simulated annealing in the presence of noise," *J. Heuristics*, 2008, doi:10.1007/s10732-007-9058-7.
61. C. Andrieu, M. Vihola, "Establishing some order amongst exact approximations of MCMCs," *Annals of Applied Probability* 26(5), 2016（本調査では未照合。記憶による）.

**データ構造・粗視化・多重レベル**

62. A. B. Bortz, M. H. Kalos, J. L. Lebowitz, "A new algorithm for Monte Carlo simulation of Ising spin systems," *J. Computational Physics* 17:10–18, 1975.
63. C. M. Fiduccia, R. M. Mattheyses, "A linear-time heuristic for improving network partitions," *Proc. 19th Design Automation Conference*, pp. 175–181, 1982.
64. R. H. Swendsen, J.-S. Wang, "Nonuniversal critical dynamics in Monte Carlo simulations," *Physical Review Letters* 58(2):86–88, 1987.
65. J. Houdayer, "A cluster Monte Carlo algorithm for 2-dimensional spin glasses," *European Physical Journal B* 22:479–484, 2001.
66. J. Houdayer, O. C. Martin, "Renormalization for discrete optimization," *Physical Review Letters* 83(5):1030–1033, 1999.
67. J. Goodman, A. D. Sokal, "Multigrid Monte Carlo method. Conceptual foundations," *Physical Review D* 40(6):2035–2071, 1989.
68. B. Hendrickson, R. Leland, "A multilevel algorithm for partitioning graphs," *Proc. Supercomputing '95*, 1995.
69. G. Karypis, V. Kumar, "A fast and high quality multilevel scheme for partitioning irregular graphs," *SIAM J. Scientific Computing* 20(1):359–392, 1998.
70. C. Walshaw, "Multilevel refinement for combinatorial optimisation problems," *Annals of Operations Research* 131:325–372, 2004.
71. J. Chen, I. Safro, "Algebraic distance on graphs," *SIAM J. Scientific Computing* 33(6):3468–3490, 2011.
72. A. Buluç, H. Meyerhenke, I. Safro, P. Sanders, C. Schulz, "Recent advances in graph partitioning," in *Algorithm Engineering*, LNCS 9220, pp. 117–158, 2016.

**スペクトル・連続緩和・平均場**

73. M. Fiedler, "Algebraic connectivity of graphs," *Czechoslovak Mathematical Journal* 23(2):298–305, 1973.
74. A. Pothen, H. D. Simon, K.-P. Liou, "Partitioning sparse matrices with eigenvectors of graphs," *SIAM J. Matrix Analysis and Applications* 11(3):430–452, 1990.
75. H. Meyerhenke, B. Monien, S. Schamberger, "Graph partitioning and disturbed diffusion," *Parallel Computing*, 2009, doi:10.1016/j.parco.2009.09.006.
76. C. Peterson, B. Söderberg, "A new method for mapping optimization problems onto neural networks," *International J. Neural Systems* 1(1):3–22, 1989.
77. D. E. Van den Bout, T. K. Miller III, "Graph partitioning using annealed neural networks," *IEEE Trans. Neural Networks* 1(2):192–203, 1990.
78. K. Rose, "Deterministic annealing for clustering, compression, classification, regression, and related optimization problems," *Proc. IEEE* 86(11):2210–2239, 1998.
79. S. Gold, A. Rangarajan, "A graduated assignment algorithm for graph matching," *IEEE Trans. PAMI* 18(4):377–388, 1996.
80. W. W. Hager, Y. Krylyuk, "Graph partitioning and continuous quadratic programming," *SIAM J. Discrete Mathematics* 12(4):500–523, 1999.

**定温MA・GPPの理論と基準**

81. M. Jerrum, G. B. Sorkin, "Simulated annealing for graph bisection," *Proc. FOCS 1993*, pp. 94–103.
82. M. Jerrum, G. B. Sorkin, "The Metropolis algorithm for graph bisection," *Discrete Applied Mathematics* 82(1–3):155–175, 1998. doi:10.1016/S0166-218X(97)00133-9
83. T. Carson, R. Impagliazzo, "Hill-climbing finds random planted bisections," *Proc. SODA 2001*, pp. 903–909.
84. M. Jerrum, "Large cliques elude the Metropolis process," *Random Structures & Algorithms* 3(4):347–359, 1992.
85. I. Wegener, "Simulated annealing beats Metropolis in combinatorial optimization," *Proc. ICALP 2005*, LNCS 3580, pp. 589–601.
86. D. T. Connolly, "An improved annealing scheme for the QAP," *European J. Operational Research* 46(1):93–100, 1990.
87. H. Cohn, M. Fielding, "Simulated annealing: Searching for an optimal temperature schedule," *SIAM J. Optimization* 9(3):779–802, 1999.
88. M. Fielding, "Simulated annealing with an optimal fixed temperature," *SIAM J. Optimization* 11(2):289–307, 2000.
89. J. E. Orosz, S. H. Jacobson, "Analysis of static simulated annealing algorithms," *J. Optimization Theory and Applications* 115:165–182, 2002.
90. A. Franzin, T. Stützle, "A landscape-based analysis of fixed temperature and simulated annealing," *European J. Operational Research* 304(2):395–410, 2023.
91. Y. Fu, P. W. Anderson, "Application of statistical mechanics to NP-complete problems in combinatorial optimisation," *J. Physics A* 19(9):1605–1620, 1986.
92. J. R. Banavar, D. Sherrington, N. Sourlas, "Graph bipartitioning and statistical mechanics," *J. Physics A* 20:L1–L8, 1987.
93. S. Boettcher, A. G. Percus, "Nature's way of optimizing," *Artificial Intelligence* 119:275–286, 2000.
94. S. Boettcher, A. G. Percus, "Optimization with extremal dynamics," *Physical Review Letters* 86:5211–5214, 2001.
95. S. Boettcher, A. G. Percus, "Extremal optimization for graph partitioning," *Physical Review E* 64:026114, 2001.
96. S. Boettcher, "Extremal optimization of graph partitioning at the percolation threshold," *J. Physics A* 32:5201–5211, 1999.
