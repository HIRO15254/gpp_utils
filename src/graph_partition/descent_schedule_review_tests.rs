//! Review instrumentation test: the cancellation checks of tracked scans
//! against the full scan's schedule (Flip: before vertex k * 1024; Swap:
//! before row k * period - 1 with period = ceil(1024 / |B|)).
use super::*;
use rand::seq::SliceRandom;

/// The events recorded since the last call, which (re)starts recording on
/// this thread.
fn take_events() -> Vec<(u8, usize)> {
    EVENTS.with(|e| e.borrow_mut().replace(Vec::new()).unwrap_or_default())
}

fn random_graph(n: usize, degree: f64, rng: &mut Mt19937GenRand64) -> Graph {
    let p = degree / (n.max(2) - 1) as f64;
    let mut edges = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            if rng.r#gen::<f64>() < p {
                edges.push([a, b]);
            }
        }
    }
    Graph::from_edges(n, edges).unwrap()
}

#[derive(Default, Debug)]
struct Worst {
    cases: u64,
    checks_old: u64,
    checks_new: u64,
    merged: u64,
    max_rows_between: usize,
    max_scored_between: usize,
    max_row_tests_between: usize,
}

fn check_swap(events: &[(u8, usize)], p: &[bool], worst: &mut Worst, context: &str) {
    let rows = p.iter().filter(|&&x| x).count();
    let row_len = p.len() - rows;
    if rows * row_len == 0 {
        assert!(events.is_empty(), "{context}");
        return;
    }
    let period = CHECK_INTERVAL.div_ceil(row_len);
    let old: Vec<usize> = (1..)
        .map(|k| k * period - 1)
        .take_while(|&r| r < rows)
        .collect();
    // Ranks of worked rows are the true ranks of their vertices.
    let rank_of: Vec<usize> = {
        let mut out = vec![usize::MAX; p.len()];
        let mut r = 0;
        for v in 0..p.len() {
            if p[v] {
                out[v] = r;
                r += 1;
            }
        }
        out
    };
    let mut last_tested = None;
    for &(kind, at) in events {
        match kind {
            3 => last_tested = Some(at),
            2 => {
                let a = last_tested.expect("a worked row follows its test");
                assert_eq!(rank_of[a], at, "rank of {a} {context}");
            }
            _ => {}
        }
    }
    let checks: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| e.0 == 0)
        .map(|(i, _)| i)
        .collect();
    worst.checks_old += old.len() as u64;
    worst.checks_new += checks.len() as u64;
    // (1) Each old check row r: some check event lies after all work on rows
    // < r and before any work on rows >= r.
    for &r in &old {
        let last_before = events
            .iter()
            .rposition(|&(k, at)| (k == 1 || k == 2) && at < r);
        let first_after = events
            .iter()
            .position(|&(k, at)| (k == 1 || k == 2) && at >= r);
        let ok = checks
            .iter()
            .any(|&i| last_before.is_none_or(|lb| i > lb) && first_after.is_none_or(|fa| i < fa));
        assert!(
            ok,
            "old check row {r} not covered: {context} period={period} rows={rows}"
        );
        if first_after.is_none_or(|fa| events[fa].1 != r) {
            worst.merged += 1;
        }
    }
    // (2) No spurious checks: each check covers an old check row not covered
    // by the previous one.
    let mut previous: Option<usize> = None;
    for &i in &checks {
        let (_, at) = events[i];
        assert!(
            old.iter()
                .any(|&r| r <= at && previous.is_none_or(|p| r > p)),
            "spurious check at {at}: {context}"
        );
        previous = Some(at);
    }
    assert!(checks.len() <= old.len(), "{context}");
    // (3) Work between checks.
    let mut bounds: Vec<usize> = checks.clone();
    bounds.push(events.len());
    let mut from = 0;
    for &to in &bounds {
        let slice = &events[from..to];
        let rows_between = slice.iter().filter(|e| e.0 == 2).count();
        let scored_between = slice.iter().filter(|e| e.0 == 1).count();
        let tests_between = slice.iter().filter(|e| e.0 == 3).count();
        worst.max_rows_between = worst.max_rows_between.max(rows_between);
        worst.max_scored_between = worst.max_scored_between.max(scored_between);
        worst.max_row_tests_between = worst.max_row_tests_between.max(tests_between);
        assert!(
            rows_between <= period,
            "{rows_between} rows between checks, period {period}: {context}"
        );
        assert!(scored_between <= period * row_len, "{context}");
        from = to + 1;
    }
}

fn check_flip(events: &[(u8, usize)], n: usize, context: &str) {
    let old: Vec<usize> = (1..)
        .map(|k| k * CHECK_INTERVAL)
        .take_while(|&v| v < n)
        .collect();
    let checks: Vec<usize> = events.iter().filter(|e| e.0 == 0).map(|e| e.1).collect();
    assert_eq!(checks, old, "{context}");
    let mut position = 0;
    for &(kind, at) in events {
        if kind == 0 {
            position = at;
        } else {
            assert!(
                at >= position && at < position + CHECK_INTERVAL,
                "{context}"
            );
        }
    }
}

#[test]
fn review_cancellation_schedule_of_tracked_scans() {
    let mut rng = Mt19937GenRand64::new(2024);
    let mut worst = Worst::default();
    let cancel = CancellationToken::new();
    for &(n, degree) in &[
        (5usize, 2.0),
        (40, 3.0),
        (100, 5.0),
        (150, 30.0),
        (300, 4.0),
        (700, 10.0),
        (1100, 3.0),
        (2100, 8.0),
        (3000, 2.0),
    ] {
        let graph = random_graph(n, degree, &mut rng);
        for trial in 0..24 {
            // |B| from 1 to n - 1, and balanced.
            let size_b = match trial % 6 {
                0 => 1,
                1 => 2.min(n - 1),
                2 => 3.min(n - 1),
                3 => n / 2,
                4 => rng.gen_range(1..n),
                _ => (n - 1).min(1025),
            };
            let mut p: Vec<bool> = (0..n).map(|v| v >= size_b).collect();
            p.shuffle(&mut rng);
            let state = PartitionState::new(&graph, p.clone()).unwrap();
            for neighborhood in [Neighborhood::Flip, Neighborhood::Swap] {
                if neighborhood == Neighborhood::Swap && n > 1100 && size_b > 3 && size_b < n - 3 {
                    // Keep the quadratic Swap scans small.
                    if trial % 3 != 0 {
                        continue;
                    }
                }
                for alpha in [0.05, 0.0, 1e17] {
                    let score = state.score(alpha);
                    for start in [score, score + 3.0, f64::INFINITY, score - 1.0] {
                        let mut scan =
                            BestImprovement::new(neighborhood, alpha, NonFinite::Compare);
                        let mut tie = Mt19937GenRand64::new(trial as u64);
                        let mut evaluations = 0;
                        take_events();
                        scan.scan(&graph, &state, start, &mut tie, &cancel, &mut evaluations)
                            .unwrap();
                        let events = take_events();
                        let context = format!(
                            "n={n} |B|={size_b} {neighborhood:?} alpha={alpha:e} start={start:e}"
                        );
                        match neighborhood {
                            Neighborhood::Flip => check_flip(&events, n, &context),
                            Neighborhood::Swap => check_swap(&events, &p, &mut worst, &context),
                        }
                        worst.cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("{worst:?}");
    assert!(worst.merged > 0 && worst.checks_new < worst.checks_old);
}
