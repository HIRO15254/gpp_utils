use crate::experiment::config::{Measurement, Schedule};
use std::collections::BTreeSet;

pub fn checkpoints(spec: &Measurement, max_steps: u64) -> Vec<u64> {
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
}
