//! EoFlip 適応度ベクトル（λ）どうしの順位・選択分布比較の指標計算。
//!
//! 中核は [`selection_probs`]: 新タイ規則（[`crate::run_executor::select_eo_rank`] の
//! 同率群平均化）のもとで、λ ベクトルと τ から各頂点の選択確率を**閉形式**で計算する。
//! タイ判定は走行時と同じくビット等値（`==`、ε なし）。
//!
//! 距離指標: Kendall τ_b（タイ対応順位相関）、全変動距離 / Jensen-Shannon 距離
//! （選択確率分布間）、下位 m 集合の加重 Jaccard。

use crate::run_executor::build_power_law_cdf;

/// λ 昇順の頂点順列（stable sort、走行時 `run_eo_flip` と同じ比較）。
pub fn sorted_order(lambdas: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..lambdas.len()).collect();
    order.sort_by(|&a, &b| lambdas[a].partial_cmp(&lambdas[b]).unwrap());
    order
}

/// λ 昇順順列上の同率群（ビット等値）を半開区間 `[s, e)` のリストで返す。
pub fn tie_groups(lambdas: &[f64], order: &[usize]) -> Vec<(usize, usize)> {
    let n = order.len();
    let mut groups = Vec::new();
    let mut s = 0usize;
    while s < n {
        let mut e = s + 1;
        while e < n && lambdas[order[e]] == lambdas[order[s]] {
            e += 1;
        }
        groups.push((s, e));
        s = e;
    }
    groups
}

/// 事前計算済みの順列・同率群・べき乗則 CDF から選択確率を計算する。
///
/// `cum` は [`build_power_law_cdf`] の出力。同率群 `[s, e)` は質量
/// `cum[e-1] − cum[s-1]` を群内で等分する（`select_eo_rank` の質量配分と厳密一致）。
pub fn selection_probs_from(
    order: &[usize],
    groups: &[(usize, usize)],
    cum: &[f64],
) -> Vec<f64> {
    let n = order.len();
    let mut probs = vec![0.0f64; n];
    for &(s, e) in groups {
        let lo = if s == 0 { 0.0 } else { cum[s - 1] };
        let hi = cum[e - 1];
        let per = (hi - lo) / (e - s) as f64;
        for pos in s..e {
            probs[order[pos]] = per;
        }
    }
    probs
}

/// λ ベクトルと τ から各頂点の選択確率を閉形式で計算する。
pub fn selection_probs(lambdas: &[f64], tau: f64) -> Vec<f64> {
    let order = sorted_order(lambdas);
    let groups = tie_groups(lambdas, &order);
    let cum = build_power_law_cdf(lambdas.len(), tau);
    selection_probs_from(&order, &groups, &cum)
}

/// 各頂点の competition 順位（同率群は位置の平均、0-indexed）。
///
/// 順位帯の解析（Δrank 回帰など）用。値の絶対水準に意味はなく、相対順位のみ使う。
pub fn midranks(order: &[usize], groups: &[(usize, usize)]) -> Vec<f64> {
    let mut ranks = vec![0.0f64; order.len()];
    for &(s, e) in groups {
        let mid = (s + e - 1) as f64 / 2.0;
        for pos in s..e {
            ranks[order[pos]] = mid;
        }
    }
    ranks
}

/// Kendall τ_b（タイ補正つき順位相関）。O(N²)。
///
/// どちらかが定数ベクトル（タイ補正の分母が 0）のときは `None`。
pub fn kendall_tau_b(a: &[f64], b: &[f64]) -> Option<f64> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < 2 {
        return None;
    }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let sa = if a[i] < a[j] {
                -1
            } else if a[i] > a[j] {
                1
            } else {
                0
            };
            let sb = if b[i] < b[j] {
                -1
            } else if b[i] > b[j] {
                1
            } else {
                0
            };
            let prod = sa * sb;
            if prod > 0 {
                concordant += 1;
            } else if prod < 0 {
                discordant += 1;
            }
        }
    }
    let n0 = (n as i64) * (n as i64 - 1) / 2;
    let tie_pairs = |v: &[f64]| -> i64 {
        let order = sorted_order(v);
        tie_groups(v, &order)
            .iter()
            .map(|&(s, e)| {
                let t = (e - s) as i64;
                t * (t - 1) / 2
            })
            .sum()
    };
    let n1 = tie_pairs(a);
    let n2 = tie_pairs(b);
    let denom = ((n0 - n1) as f64) * ((n0 - n2) as f64);
    if denom <= 0.0 {
        return None;
    }
    Some((concordant - discordant) as f64 / denom.sqrt())
}

/// 全変動距離 `0.5·Σ|p−q|`（∈ [0,1]）。「同じ状態で異なる頂点を選ぶ確率」の下限。
pub fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>()
}

/// Shannon エントロピー（bit）。`0·log0 := 0`。
pub fn shannon_entropy(p: &[f64]) -> f64 {
    -p.iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x * x.log2())
        .sum::<f64>()
}

/// Jensen-Shannon ダイバージェンス（base 2、∈ [0,1]）。
pub fn jensen_shannon(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    let mut jsd = 0.0f64;
    for (&a, &b) in p.iter().zip(q) {
        let m = 0.5 * (a + b);
        if a > 0.0 {
            jsd += 0.5 * a * (a / m).log2();
        }
        if b > 0.0 {
            jsd += 0.5 * b * (b / m).log2();
        }
    }
    jsd.clamp(0.0, 1.0)
}

/// 下位 m 集合の加重（Ruzicka）Jaccard。
///
/// 順位 `m` 位まで（昇順位置 `< m`）に「どれだけ属するか」を頂点ごとの重みにする:
/// 完全に下位 m 内の同率群はメンバー重み 1、境界を跨ぐ群 `[s, e)`（`s < m < e`）は
/// メンバー重み `(m − s)/(e − s)`（群内の並びは確率的に等価なので按分）。
/// `J = Σ_v min(w_a, w_b) / Σ_v max(w_a, w_b)`。タイが無ければ通常の Jaccard に一致。
/// `m ≥ n` では 1.0。
pub fn bottom_m_jaccard(
    order_a: &[usize],
    groups_a: &[(usize, usize)],
    order_b: &[usize],
    groups_b: &[(usize, usize)],
    m: usize,
) -> f64 {
    let n = order_a.len();
    if m >= n {
        return 1.0;
    }
    let weights = |order: &[usize], groups: &[(usize, usize)]| -> Vec<f64> {
        let mut w = vec![0.0f64; n];
        for &(s, e) in groups {
            let val = if e <= m {
                1.0
            } else if s < m {
                (m - s) as f64 / (e - s) as f64
            } else {
                break;
            };
            for pos in s..e {
                w[order[pos]] = val;
            }
        }
        w
    };
    let wa = weights(order_a, groups_a);
    let wb = weights(order_b, groups_b);
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for v in 0..n {
        num += wa[v].min(wb[v]);
        den += wa[v].max(wb[v]);
    }
    if den == 0.0 {
        1.0
    } else {
        num / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_executor::select_eo_rank;

    #[test]
    fn test_selection_probs_matches_select_eo_rank_empirically() {
        let lambdas = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 5.0, 9.0];
        for tau in [0.8, 1.7] {
            let order = sorted_order(&lambdas);
            let cum = build_power_law_cdf(lambdas.len(), tau);
            let probs = selection_probs(&lambdas, tau);
            // 決定的な u グリッドで select_eo_rank を掃引し、頻度と閉形式を比較。
            let trials = 200_000usize;
            let mut counts = vec![0usize; lambdas.len()];
            for i in 0..trials {
                let u = (i as f64 + 0.5) / trials as f64;
                let (v, _) = select_eo_rank(&lambdas, &order, &cum, u);
                counts[v] += 1;
            }
            for v in 0..lambdas.len() {
                let emp = counts[v] as f64 / trials as f64;
                assert!(
                    (emp - probs[v]).abs() < 2.0 / trials as f64 + 1e-9,
                    "tau={tau} v={v}: emp={emp} closed={}",
                    probs[v]
                );
            }
            let total: f64 = probs.iter().sum();
            assert!((total - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_selection_probs_no_ties_equals_power_law() {
        let lambdas: Vec<f64> = (0..20).map(|i| i as f64 * 0.37 + 0.1).collect();
        let tau = 1.4;
        let probs = selection_probs(&lambdas, tau);
        let z: f64 = (1..=20).map(|k| (k as f64).powf(-tau)).sum();
        for (k, &v) in sorted_order(&lambdas).iter().enumerate() {
            let expect = ((k + 1) as f64).powf(-tau) / z;
            assert!((probs[v] - expect).abs() < 1e-12, "k={k}");
        }
    }

    #[test]
    fn test_selection_probs_all_tied_is_uniform() {
        let lambdas = [0.0; 16];
        let probs = selection_probs(&lambdas, 1.4);
        for &p in &probs {
            assert!((p - 1.0 / 16.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_kendall_tau_b_known() {
        let a: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let rev: Vec<f64> = (0..10).map(|i| (9 - i) as f64).collect();
        assert!((kendall_tau_b(&a, &a).unwrap() - 1.0).abs() < 1e-12);
        assert!((kendall_tau_b(&a, &rev).unwrap() + 1.0).abs() < 1e-12);
        // 定数ベクトル → None。
        assert!(kendall_tau_b(&a, &[0.0; 10]).is_none());
        // タイあり手計算例: C=7, D=1, n1=n2=1 → τ_b = 6/9 = 2/3
        // （scipy.stats.kendalltau([1,2,2,3,4],[1,3,2,2,4]) と一致）。
        let x = [1.0, 2.0, 2.0, 3.0, 4.0];
        let y = [1.0, 3.0, 2.0, 2.0, 4.0];
        assert!((kendall_tau_b(&x, &y).unwrap() - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_tv_jsd_identity_and_bounds() {
        let p = [0.5, 0.3, 0.2, 0.0];
        let q = [0.0, 0.0, 0.0, 1.0];
        assert!(total_variation(&p, &p).abs() < 1e-12);
        assert!(jensen_shannon(&p, &p).abs() < 1e-12);
        // 台が交わらない → TV = 1, JSD = 1。
        assert!((total_variation(&p, &q) - 1.0).abs() < 1e-12);
        assert!((jensen_shannon(&p, &q) - 1.0).abs() < 1e-12);
        // 対称性。
        assert!((jensen_shannon(&p, &q) - jensen_shannon(&q, &p)).abs() < 1e-15);
        // エントロピー: 一様 4 点 = 2 bit。
        assert!((shannon_entropy(&[0.25; 4]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_bottom_m_jaccard_fractional() {
        // a: 明確な順序 0<1<2<3。b: 先頭 3 頂点が同率 λ=0、残り 1。
        let la = [0.0, 1.0, 2.0, 3.0];
        let lb = [0.0, 0.0, 0.0, 1.0];
        let oa = sorted_order(&la);
        let ga = tie_groups(&la, &oa);
        let ob = sorted_order(&lb);
        let gb = tie_groups(&lb, &ob);
        // m=1: a は頂点0 に重み1。b は同率群 [0,3) を跨ぐので各 1/3。
        // min 和 = 1/3, max 和 = 1 + 1/3 + 1/3 = 5/3 → J = 0.2。
        let j = bottom_m_jaccard(&oa, &ga, &ob, &gb, 1);
        assert!((j - 0.2).abs() < 1e-12, "j={j}");
        // m >= n → 1。
        assert!((bottom_m_jaccard(&oa, &ga, &ob, &gb, 4) - 1.0).abs() < 1e-12);
        // 完全一致同士。
        assert!((bottom_m_jaccard(&oa, &ga, &oa, &ga, 2) - 1.0).abs() < 1e-12);
        // 交わらない下位集合 → 0。
        let lc = [3.0, 2.0, 1.0, 0.0];
        let oc = sorted_order(&lc);
        let gc = tie_groups(&lc, &oc);
        assert!(bottom_m_jaccard(&oa, &ga, &oc, &gc, 1).abs() < 1e-12);
    }
}
