use crate::experiment::config::{Measurement, Schedule};
use std::cell::RefCell;
use std::collections::BTreeSet;

thread_local! {
    // Only the last logarithmic budget is retained. Its list has at most 181
    // entries even for u64::MAX, independent of the number of experiments.
    static LAST_LOGARITHMIC: RefCell<Option<(u64, Vec<u64>)>> = const { RefCell::new(None) };
}

pub fn checkpoints(spec: &Measurement, max_steps: u64) -> Vec<u64> {
    if spec.schedule == Schedule::Logarithmic {
        return LAST_LOGARITHMIC.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some((budget, points)) = &*cache
                && *budget == max_steps
            {
                return points.clone();
            }
            let points = calculate_checkpoints(spec, max_steps);
            *cache = Some((max_steps, points.clone()));
            points
        });
    }
    calculate_checkpoints(spec, max_steps)
}

fn calculate_checkpoints(spec: &Measurement, max_steps: u64) -> Vec<u64> {
    let mut out = BTreeSet::from([0, max_steps]);
    match spec.schedule {
        Schedule::Explicit => out.extend(spec.steps.iter().copied().filter(|&x| x <= max_steps)),
        Schedule::Logarithmic => {
            let mut scale = 1u64;
            loop {
                for digit in 1..=9 {
                    if let Some(x) = scale.checked_mul(digit) {
                        if x > max_steps {
                            return out.into_iter().collect();
                        }
                        out.insert(x);
                    } else {
                        return out.into_iter().collect();
                    }
                }
                match scale.checked_mul(10) {
                    Some(x) => scale = x,
                    None => break,
                }
            }
        }
    }
    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::config::BasinMode;
    #[test]
    fn logarithmic() {
        let m = Measurement {
            schedule: Schedule::Logarithmic,
            steps: vec![],
            basin: BasinMode::None,
            max_basin_steps: 1,
            diagnostics: false,
            best_basin: false,
        };
        assert_eq!(
            checkpoints(&m, 12),
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
        );
    }

    #[test]
    fn cached_logarithmic_budgets_and_explicit_schedules_stay_exact() {
        let mut spec = Measurement::default();
        for budget in [0, 1, 9, 10, 12, 1_000_000, u64::MAX, 1_000_000] {
            let expected = calculate_checkpoints(&spec, budget);
            assert_eq!(checkpoints(&spec, budget), expected);
            spec.diagnostics = !spec.diagnostics;
            assert_eq!(checkpoints(&spec, budget), expected);
        }
        spec.schedule = Schedule::Explicit;
        spec.steps = vec![9, 3, 3, 200];
        assert_eq!(checkpoints(&spec, 10), vec![0, 3, 9, 10]);
        spec.steps = vec![1, 7];
        assert_eq!(checkpoints(&spec, 10), vec![0, 1, 7, 10]);
    }
}
