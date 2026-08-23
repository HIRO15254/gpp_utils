//! EO のランク抽選（[`crate::run_executor::select_eo_rank`]）を**差分更新**で行うための索引。
//!
//! # 何を速くするか
//!
//! τ-EO は毎ステップ「全頂点の適応度 λ を作る → λ 昇順に全体ソート → べき乗則でランクを引く」を
//! 行う。これは O(N log N)/step で、EO の実行時間のほぼ全部を占める。
//!
//! しかし元論文の適応度 λ_v = g_v/deg_v = `(deg_v − cuts_v)/deg_v` は
//! **`cuts_v` だけの関数**（deg は不変）であり、1 頂点をフリップしたとき λ が変わるのは
//! **その頂点とその隣接だけ**（1 + deg 個）。したがって順位構造を差分更新できる。
//!
//! # 構造
//!
//! λ の取りうる値は事前に確定する（グラフに現れる各 degree `d` について `(d−c)/d`, `c = 0..=d`）。
//! これを昇順・重複排除して**スロット**とし、
//!
//! - `members[slot]`: そのスロットに居る頂点番号（**昇順**を維持）
//! - `fenwick`: スロット別頂点数の Fenwick 木（前置和とランク検索が O(log M)）
//!
//! を保つ。1 頂点の λ 変化は「旧スロットから除去 → 新スロットへ挿入」の O(log M + 群長) で済む。
//!
//! # ソート版との厳密一致
//!
//! [`crate::run_executor::select_eo_rank`] は「先頭から同率ブロックを見て `u < cum[e−1]` となる
//! 最初のブロック」を選ぶ。`cum` は単調増加なので、これは
//! **`p = min{k : cum[k] > u}`（無ければ `n−1`）を含むブロック**と同値である。
//! ブロック内の順序は「恒等列 + 安定ソート」＝頂点番号昇順なので、`members` を昇順に保てば
//! *選ばれる頂点まで含めて* ビット完全一致する。群の判定はソート版と同じ `f64` の `==` で行う。
//!
//! # 適用範囲
//!
//! λ が `(deg, cuts)` だけで決まる場合に限る。すなわち
//! [`crate::run_config::SolverSpec::Eo`]（スワップ版）と
//! [`crate::run_config::SolverSpec::EoFlipMulAlpha`] の `alpha == 1.0`（λ1 ≡ 1 なので
//! `swap_fitness * 1.0` は f64 恒等）だけ。多数派/少数派やバランスペナルティに依存する
//! 適応度（Legacy / AddBeta / MulGamma / α≠1）は集合サイズが動くと全頂点の λ が同時に変わるため、
//! 従来のソート経路を使うこと。

use crate::run_executor::swap_fitness;

/// λ = g/deg 専用のランク索引。
#[derive(Debug, Clone)]
pub struct EoRankIndex {
    /// 昇順・重複排除済みの相異なる λ 値（長さ M）。
    slot_value: Vec<f64>,
    /// degree `d` の LUT 開始位置（`d` がグラフに現れないときは [`u32::MAX`]）。
    deg_offset: Vec<u32>,
    /// `lut[deg_offset[d] + c]` = λ = (d−c)/d のスロット id。
    lut: Vec<u32>,
    /// スロットごとの在籍頂点（昇順）。
    members: Vec<Vec<u32>>,
    /// スロット別在籍数の Fenwick 木（1-indexed、長さ M+1）。
    fenwick: Vec<u32>,
    /// `fenwick` 探索用の最上位 2 冪。
    top_bit: usize,
    /// 頂点 → 現在のスロット。
    slot_of: Vec<u32>,
    /// 頂点数。
    n: usize,
}

impl EoRankIndex {
    /// `degrees`（各頂点の次数）と `cuts_at`（各頂点のカット辺数）から索引を構築する。O(N + M)。
    pub fn new(degrees: &[usize], cuts_at: &[i32]) -> Self {
        let n = degrees.len();
        assert_eq!(n, cuts_at.len(), "degrees と cuts_at の長さが不一致");
        let max_deg = degrees.iter().copied().max().unwrap_or(0);

        let mut present = vec![false; max_deg + 1];
        for &d in degrees {
            present[d] = true;
        }

        // λ の候補値をすべて集めて昇順・重複排除する。
        // 重複判定はソート版の同率判定（`lambdas[a] == lambdas[b]`）と同じ `==` を使う。
        let mut values: Vec<f64> = Vec::new();
        for (d, &p) in present.iter().enumerate() {
            if !p {
                continue;
            }
            for c in 0..=d {
                values.push(swap_fitness(d, c as i32));
            }
        }
        values.sort_by(|a, b| a.partial_cmp(b).expect("λ に NaN は現れない"));
        values.dedup();
        let slot_value = values;
        let m = slot_value.len();
        assert!(m > 0 || n == 0, "スロットが空");

        // (deg, cuts) → slot の平坦化 LUT。存在する degree の分だけ確保する。
        let mut deg_offset = vec![u32::MAX; max_deg + 1];
        let mut lut: Vec<u32> = Vec::new();
        for (d, &p) in present.iter().enumerate() {
            if !p {
                continue;
            }
            deg_offset[d] = lut.len() as u32;
            for c in 0..=d {
                let lam = swap_fitness(d, c as i32);
                let s = slot_value
                    .binary_search_by(|probe| probe.partial_cmp(&lam).expect("NaN なし"))
                    .expect("候補値は必ずスロットに存在する");
                lut.push(s as u32);
            }
        }

        let mut members: Vec<Vec<u32>> = vec![Vec::new(); m];
        let mut slot_of = vec![0u32; n];
        for v in 0..n {
            let s = lut[deg_offset[degrees[v]] as usize + cuts_at[v] as usize];
            slot_of[v] = s;
            // v の昇順ループなので push だけで `members` は昇順になる。
            members[s as usize].push(v as u32);
        }

        let mut fenwick = vec![0u32; m + 1];
        for (s, mem) in members.iter().enumerate() {
            fenwick[s + 1] = mem.len() as u32;
        }
        for i in 1..=m {
            let j = i + (i & i.wrapping_neg());
            if j <= m {
                fenwick[j] += fenwick[i];
            }
        }

        let mut top_bit = 1usize;
        while top_bit * 2 <= m {
            top_bit *= 2;
        }

        Self { slot_value, deg_offset, lut, members, fenwick, top_bit, slot_of, n }
    }

    /// 頂点 `v` の `cuts` が `new_cuts` になったことを反映する。O(log M + 群長)。
    ///
    /// λ が変わらない（同じスロットのまま）ときは何もしないので、同じ頂点に対する
    /// 冪等な呼び出し（多重辺・重複隣接）でも安全。
    pub fn update_vertex(&mut self, v: usize, deg: usize, new_cuts: i32) {
        let new_slot = self.slot_for(deg, new_cuts);
        let old_slot = self.slot_of[v] as usize;
        if old_slot == new_slot {
            return;
        }
        let vv = v as u32;

        let pos = self.members[old_slot]
            .binary_search(&vv)
            .expect("頂点は旧スロットに在籍しているはず");
        self.members[old_slot].remove(pos);
        self.fenwick_add(old_slot, false);

        let ins = self.members[new_slot].partition_point(|&x| x < vv);
        self.members[new_slot].insert(ins, vv);
        self.fenwick_add(new_slot, true);

        self.slot_of[v] = new_slot as u32;
    }

    /// べき乗則ランク抽選。[`crate::run_executor::select_eo_rank`] と**返り値が厳密一致**する。
    ///
    /// `cum` は [`crate::run_executor::build_power_law_cdf`] の出力、`u ∈ [0,1)` は一様乱数。
    /// 戻り値は `(選択頂点, その昇順位置 0-indexed)`。
    pub fn select(&self, cum: &[f64], u: f64) -> (usize, usize) {
        let n = self.n;
        debug_assert!(n > 0);
        // ソート版の「先頭から `u < cum[e−1]` となる最初の同率ブロック」は、
        // `cum` が単調増加なので「`p = min{k : cum[k] > u}` を含むブロック」と同値。
        // 丸め誤差で u ≥ cum[n−1] になったときは最終ブロック（= 位置 n−1）に落とす。
        let p = cum.partition_point(|&c| c <= u).min(n - 1);
        let (slot, s) = self.fenwick_find(p as u32);
        let s = s as usize;
        let m = self.members[slot].len();
        let e = s + m;
        let lo = if s == 0 { 0.0 } else { cum[s - 1] };
        let hi = cum[e - 1];
        let per = (hi - lo) / m as f64;
        let off = if per > 0.0 {
            (((u - lo).max(0.0) / per) as usize).min(m - 1)
        } else {
            0
        };
        (self.members[slot][off] as usize, s + off)
    }

    /// `(deg, cuts)` に対応するスロット id。
    #[inline]
    fn slot_for(&self, deg: usize, cuts: i32) -> usize {
        self.lut[self.deg_offset[deg] as usize + cuts as usize] as usize
    }

    /// Fenwick 木に ±1 する。
    #[inline]
    fn fenwick_add(&mut self, slot: usize, inc: bool) {
        let m = self.fenwick.len() - 1;
        let mut i = slot + 1;
        while i <= m {
            if inc {
                self.fenwick[i] += 1;
            } else {
                self.fenwick[i] -= 1;
            }
            i += i & i.wrapping_neg();
        }
    }

    /// 0-indexed の順位 `p` を含むスロットと、そのスロットの先頭順位を返す。O(log M)。
    #[inline]
    fn fenwick_find(&self, p: u32) -> (usize, u32) {
        let m = self.fenwick.len() - 1;
        let mut idx = 0usize;
        let mut rem = p;
        let mut bit = self.top_bit;
        while bit > 0 {
            let next = idx + bit;
            if next <= m && self.fenwick[next] <= rem {
                idx = next;
                rem -= self.fenwick[next];
            }
            bit >>= 1;
        }
        // idx は prefix(idx) ≤ p を満たす最大の位置なので、0-indexed スロット idx の
        // 在籍数は必ず 1 以上（0 なら prefix が伸びず idx が最大にならない）。
        (idx, p - rem)
    }

    /// 索引の内部整合性を検証する（デバッグ用）。
    pub fn debug_verify(&self, degrees: &[usize], cuts_at: &[i32]) -> Result<(), String> {
        let m = self.members.len();
        for (v, (&d, &c)) in degrees.iter().zip(cuts_at.iter()).enumerate() {
            let want = self.lut[self.deg_offset[d] as usize + c as usize];
            if self.slot_of[v] != want {
                return Err(format!("slot_of[{v}] = {} だが期待は {want}", self.slot_of[v]));
            }
        }
        let mut total = 0usize;
        for s in 0..m {
            let mem = &self.members[s];
            if mem.windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!("members[{s}] が昇順でない"));
            }
            for &v in mem {
                if self.slot_of[v as usize] as usize != s {
                    return Err(format!("members[{s}] に他スロットの頂点 {v}"));
                }
            }
            total += mem.len();
            let prefix: usize = (0..=s).map(|k| self.members[k].len()).sum();
            if self.fenwick_prefix(s + 1) as usize != prefix {
                return Err(format!("fenwick 前置和が不一致 (slot {s})"));
            }
        }
        if total != self.n {
            return Err(format!("在籍総数 {total} ≠ 頂点数 {}", self.n));
        }
        Ok(())
    }

    /// 1-indexed 位置 `i` までの前置和（検証用）。
    fn fenwick_prefix(&self, i: usize) -> u32 {
        let mut s = 0u32;
        let mut k = i;
        while k > 0 {
            s += self.fenwick[k];
            k -= k & k.wrapping_neg();
        }
        s
    }

    /// スロット値（テスト用）。
    #[cfg(test)]
    pub(crate) fn slot_values(&self) -> &[f64] {
        &self.slot_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_executor::{build_power_law_cdf, select_eo_rank};

    /// `degrees`/`cuts_at` から素朴に λ と昇順 order を作る（ソート版と同じ手順）。
    fn naive(degrees: &[usize], cuts_at: &[i32]) -> (Vec<f64>, Vec<usize>) {
        let n = degrees.len();
        let lambdas: Vec<f64> =
            (0..n).map(|v| swap_fitness(degrees[v], cuts_at[v])).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| lambdas[a].partial_cmp(&lambdas[b]).unwrap());
        (lambdas, order)
    }

    /// スロット値は昇順・重複なし。
    #[test]
    fn test_slots_sorted_and_unique() {
        let degrees = vec![4, 4, 2, 6, 0, 3];
        let cuts_at = vec![0, 2, 1, 3, 0, 3];
        let ix = EoRankIndex::new(&degrees, &cuts_at);
        let vals = ix.slot_values();
        assert!(vals.windows(2).all(|w| w[0] < w[1]), "スロット値が昇順・一意でない");
        // deg=4 の 1/2 と deg=2 の 1/2、deg=6 の 3/6 は同一スロットに畳まれる。
        assert!(vals.contains(&0.5));
    }

    /// 構築直後の `select` がソート版と全 u で一致する。
    #[test]
    fn test_select_matches_sort_version() {
        let degrees = vec![4, 4, 2, 6, 0, 3, 5, 5];
        let cuts_at = vec![0, 2, 1, 3, 0, 3, 5, 0];
        let ix = EoRankIndex::new(&degrees, &cuts_at);
        let (lambdas, order) = naive(&degrees, &cuts_at);
        for tau in [0.0_f64, 0.6, 1.3, 2.0] {
            let cum = build_power_law_cdf(degrees.len(), tau);
            for k in 0..20_000 {
                let u = k as f64 / 20_000.0;
                let want = select_eo_rank(&lambdas, &order, &cum, u);
                let got = ix.select(&cum, u);
                assert_eq!(want, got, "tau={tau}, u={u}");
            }
            // 丸め誤差で u ≥ cum[n−1] になる場合のフォールバック。
            let u = 1.0 + 1e-12;
            assert_eq!(select_eo_rank(&lambdas, &order, &cum, u), ix.select(&cum, u));
        }
    }

    /// ランダムな `cuts` 更新を繰り返しても索引がソート版と一致し続ける。
    #[test]
    fn test_updates_stay_consistent() {
        let degrees = vec![3, 5, 5, 2, 4, 4, 6, 1, 0, 7];
        let mut cuts_at: Vec<i32> = degrees.iter().map(|&d| (d / 2) as i32).collect();
        let mut ix = EoRankIndex::new(&degrees, &cuts_at);
        let cum = build_power_law_cdf(degrees.len(), 1.4);

        // 決定的な擬似乱数で頂点と新 cuts を選ぶ。
        let mut state = 0x1234_5678_u64;
        for _ in 0..2_000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = (state >> 33) as usize % degrees.len();
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let c = ((state >> 33) as usize % (degrees[v] + 1)) as i32;

            cuts_at[v] = c;
            ix.update_vertex(v, degrees[v], c);
            ix.debug_verify(&degrees, &cuts_at).expect("索引の整合性");

            let (lambdas, order) = naive(&degrees, &cuts_at);
            for k in 0..200 {
                let u = k as f64 / 200.0;
                assert_eq!(
                    select_eo_rank(&lambdas, &order, &cum, u),
                    ix.select(&cum, u),
                    "u={u}, cuts_at={cuts_at:?}"
                );
            }
        }
    }

    /// 全頂点が同率（同じ λ）のときも一致する。
    #[test]
    fn test_all_tied() {
        let degrees = vec![4; 8];
        let cuts_at = vec![2; 8];
        let ix = EoRankIndex::new(&degrees, &cuts_at);
        let (lambdas, order) = naive(&degrees, &cuts_at);
        let cum = build_power_law_cdf(8, 1.4);
        for k in 0..5_000 {
            let u = k as f64 / 5_000.0;
            assert_eq!(select_eo_rank(&lambdas, &order, &cum, u), ix.select(&cum, u));
        }
    }
}
