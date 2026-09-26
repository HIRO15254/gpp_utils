//! Exact memo of the Metropolis acceptance factor `(-delta / t).exp()`.
//!
//! SA and EO-SA judge an uphill or tied move by drawing `u` and comparing it
//! with `(-delta / t).exp()` for the job's fixed temperature `t`. The real
//! objective takes few distinct values of `delta`, so the same factor is
//! evaluated again and again. [`MetropolisFactor`] stores the evaluated factor
//! of recent `delta` bit patterns; it never replaces the expression by another
//! formula and never touches a random number generator.

/// log2 of the number of entries: 1024 entries of 16 bytes.
const BITS: u32 = 10;
const LEN: usize = 1 << BITS;

/// The factor as SA and EO-SA evaluate it.
#[inline]
fn factor(delta: f64, t: f64) -> f64 {
    (-delta / t).exp()
}

/// Direct-mapped memo of `(-delta / t).exp()` for one fixed `t`, keyed by
/// `delta.to_bits()`.
///
/// Invariant: every entry `(key, value)` holds `value == factor(f64::from_bits(key), t)`
/// with the same bits, because `value` is the result of that expression on
/// exactly those inputs. The initial entries hold the key of `+0.0` and its
/// value; a miss evaluates the expression and overwrites one entry. So
/// [`Self::get`] returns the bits that evaluating the expression would return,
/// for every `delta` including zeros, infinities and NaN payloads.
#[derive(Clone, Debug)]
pub(crate) struct MetropolisFactor {
    t: f64,
    entries: Box<[(u64, f64); LEN]>,
}

impl MetropolisFactor {
    pub(crate) fn new(t: f64) -> Self {
        let zero = 0.0f64;
        let entry = (zero.to_bits(), factor(zero, t));
        Self {
            t,
            entries: vec![entry; LEN]
                .into_boxed_slice()
                .try_into()
                .expect("LEN entries"),
        }
    }

    /// `(-delta / t).exp()`, bit for bit.
    #[inline]
    pub(crate) fn get(&mut self, delta: f64) -> f64 {
        let key = delta.to_bits();
        // Fibonacci hashing; the shift keeps the slot below `LEN`.
        let slot = (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (u64::BITS - BITS)) as usize;
        let entry = &mut self.entries[slot];
        if entry.0 == key {
            entry.1
        } else {
            let value = factor(delta, self.t);
            *entry = (key, value);
            value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand_mt::Mt19937GenRand64;
    use std::hint::black_box;

    fn assert_exact(memo: &mut MetropolisFactor, delta: f64, t: f64) {
        // `black_box` keeps the expected value a run-time evaluation.
        let expected = (-black_box(delta) / black_box(t)).exp();
        assert_eq!(
            memo.get(delta).to_bits(),
            expected.to_bits(),
            "delta {delta:e} ({:#x}) t {t:e}",
            delta.to_bits()
        );
    }

    /// Hits, misses, collisions and replacements return the bits of the
    /// expression, for positive, zero, subnormal, huge and non-finite
    /// temperatures and every class of `delta`.
    #[test]
    fn memo_returns_the_bits_of_the_expression() {
        let special = [
            0.0,
            -0.0,
            5e-324,
            -5e-324,
            f64::MIN_POSITIVE,
            1e-300,
            0.05,
            0.2,
            1.0,
            1.0 + f64::EPSILON,
            2.0,
            3.5,
            700.0,
            710.0,
            746.0,
            1e300,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7ff0_0000_0000_0001),
            -1.0,
            -700.0,
        ];
        for t in [
            1.0,
            0.03162277660168379,
            316.22776601683796,
            1e-300,
            5e-324,
            1e300,
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NAN,
        ] {
            let mut memo = MetropolisFactor::new(t);
            let mut rng = Mt19937GenRand64::new(t.to_bits());
            // Scores like those of the real objective: integers plus rounded
            // balance penalties, so many deltas repeat and some collide.
            let values: Vec<f64> = (0..3000)
                .map(|i| {
                    let d = (i % 61) as f64 - 30.0;
                    (500 + i / 61) as f64 + 0.05 * d * d
                })
                .collect();
            for _ in 0..20_000 {
                let a = values[rng.gen_range(0..values.len())];
                let b = values[rng.gen_range(0..values.len())];
                assert_exact(&mut memo, a - b, t);
                if rng.gen_range(0..16) == 0 {
                    let x = special[rng.gen_range(0..special.len())];
                    assert_exact(&mut memo, x, t);
                }
                if rng.gen_range(0..64) == 0 {
                    assert_exact(&mut memo, f64::from_bits(rng.r#gen()), t);
                }
            }
            for &x in &special {
                assert_exact(&mut memo, x, t);
                assert_exact(&mut memo, x, t);
            }
        }
    }

    /// Keys that share a slot evict each other and are evaluated again.
    #[test]
    fn colliding_keys_stay_exact() {
        let t = 0.7;
        let mut memo = MetropolisFactor::new(t);
        let slot =
            |x: f64| (x.to_bits().wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - BITS)) as usize;
        let first = 1.25f64;
        let second = (1..)
            .map(|i| first + i as f64 * 0.125)
            .find(|&x| slot(x) == slot(first))
            .unwrap();
        for _ in 0..4 {
            for x in [first, second, 0.0, -0.0] {
                assert_exact(&mut memo, x, t);
            }
        }
    }
}
