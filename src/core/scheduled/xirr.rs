use super::{year_fraction, DayCount};
use crate::core::{
    models::{validate, validate_length, DateLike, InvalidPaymentsError},
    optimize::brentq,
    utils::{fast_pow, initial_guess},
};

pub fn xirr(
    dates: &[DateLike],
    amounts: &[f64],
    guess: Option<f64>,
    day_count: Option<DayCount>,
) -> Result<f64, InvalidPaymentsError> {
    validate(amounts, Some(dates))?;

    let deltas = &day_count_factor(dates, day_count);

    if amounts.len() == 2 {
        return Ok(xirr_analytical_2(amounts, deltas));
    }

    // Scaling all payments leaves the roots unchanged and makes the solver's
    // residual tolerances independent of the currency units.
    let scale = amounts.iter().map(|amount| amount.abs()).fold(0.0, f64::max);
    let amounts = &amounts.iter().map(|amount| amount / scale).collect::<Vec<_>>();

    let f = |rate| xnpv_result(amounts, deltas, rate);
    let fd = |rate| xnpv_result_with_deriv(amounts, deltas, rate);

    let guess = guess.unwrap_or_else(|| initial_guess(amounts));
    let rate = newton_raphson(guess, &fd);

    if rate.is_finite() {
        return Ok(rate);
    }

    let rate = brentq(&f, -0.999999999999999, 100., 100);

    if rate.is_finite() {
        return Ok(rate);
    }

    let mut step = 0.01;
    let mut guess = -0.99999999999999;
    while guess < 1.0 {
        let rate = newton_raphson(guess, &fd);
        if rate.is_finite() {
            return Ok(rate);
        }
        guess += step;
        step = (step * 1.1).min(0.1);
    }

    Ok(f64::NAN)
}

// XIRR rates must be above -1. Keep this domain policy beside XNPV.
fn newton_raphson<Func>(start: f64, fd: &Func) -> f64
where
    Func: Fn(f64) -> (f64, f64),
{
    const RATE_TOL: f64 = 1e-9;
    let mut x = start;
    for _ in 0..50 {
        let (value, derivative) = fd(x);
        if value == 0.0 {
            return x;
        }
        let delta = value / derivative;
        if delta.abs() < RATE_TOL && value.abs() < 1e-3 {
            return x - delta;
        }
        let next = x - delta;
        if next == x {
            // A steep root can have a large residual at the nearest float.
            // Probe within the rate tolerance, or representable spacing.
            let step = RATE_TOL.max(f64::EPSILON * x.abs());
            let lo = x - step.min((1.0 + x) / 2.0);
            let hi = x + step;
            if lo > -1.0 && lo < x && hi > x && hi.is_finite() {
                let (f_lo, f_hi) = (fd(lo).0, fd(hi).0);
                let clear = |v: f64| v.is_finite() && v.abs() > value.abs();
                // ponytail: sampled signs are a heuristic, not an error bound;
                // certified root guarantees would require interval evaluation.
                if clear(f_lo) && clear(f_hi) && (f_lo < 0.0) != (f_hi < 0.0) {
                    return x;
                }
            }
            return f64::NAN;
        }
        // At large rates, evaluation noise can leave Newton cycling above
        // RATE_TOL. Refine only inside a narrow, sign-qualified neighbourhood.
        let step = 1e-12 * x.abs();
        if delta.abs() < step && value.abs() < 1e-3 {
            let lo = x - step.min((1.0 + x) / 2.0);
            let hi = x + step;
            if lo > -1.0 && lo < x && hi > x && hi.is_finite() {
                let (f_lo, f_hi) = (fd(lo).0, fd(hi).0);
                let clear = |v: f64| v.is_finite() && v.abs() > value.abs();
                // The stall check's sampled-sign heuristic applies here too.
                if clear(f_lo) && clear(f_hi) && (f_lo < 0.0) != (f_hi < 0.0) {
                    let rate = brentq(&|r| fd(r).0, lo, hi, 100);
                    if rate.is_finite()
                        && rate > -1.0
                        && rate >= lo
                        && rate <= hi
                        && fd(rate).0.abs() < 1e-3
                    {
                        return rate;
                    }
                }
            }
        }
        x = next;
    }
    f64::NAN
}

fn xirr_analytical_2(amounts: &[f64], deltas: &[f64]) -> f64 {
    // solve analytically:
    // cf[0]/(1+r)^d[0] + cf[1]/(1+r)^d[1] = 0  =>
    // cf[1]/(1+r)^d[1] = -cf[0]/(1+r)^d[0]  => rearrange
    // cf[1]/cf[0] = -(1+r)^d[1]/(1+r)^d[0]  => simplify
    // cf[1]/cf[0] = -(1+r)^(d[1] - d[0])  => take the root
    // (cf[1]/cf[0])^(1/(d[1] - d[0])) = -(1 + r) => multiply by -1 and subtract 1
    // r = -(cf[1]/cf[0])^(1/(d[1] - d[0])) - 1
    (-amounts[1] / amounts[0]).powf(1. / (deltas[1] - deltas[0])) - 1.0
}

/// Calculate the net present value of a series of payments at irregular intervals.
pub fn xnpv(
    rate: f64,
    dates: &[DateLike],
    amounts: &[f64],
    day_count: Option<DayCount>,
) -> Result<f64, InvalidPaymentsError> {
    validate_length(amounts, dates)?;

    let deltas = &day_count_factor(dates, day_count);
    Ok(xnpv_result(amounts, deltas, rate))
}

pub fn sign_changes(v: &[f64]) -> i32 {
    v.windows(2)
        .map(|p| (p[0].is_finite() && p[1].is_finite() && p[0].signum() != p[1].signum()) as i32)
        .sum()
}

pub fn zero_crossing_points(v: &[f64]) -> Vec<usize> {
    v.windows(2)
        .enumerate()
        .filter_map(|(i, p)| {
            (p[0].is_finite() && p[1].is_finite() && p[0].signum() != p[1].signum()).then_some(i)
        })
        .collect()
}

fn day_count_factor(dates: &[DateLike], day_count: Option<DayCount>) -> Vec<f64> {
    let min_date = dates.iter().min().unwrap();
    let dc = day_count.unwrap_or_default();
    dates.iter().map(|d| year_fraction(&min_date, &d, dc)).collect()
}

// \sum_{i=1}^n \frac{P_i}{(1 + rate)^{(d_i - d_0)/365}}
fn xnpv_result(payments: &[f64], deltas: &[f64], rate: f64) -> f64 {
    if rate <= -1.0 {
        // bound newton_raphson
        return f64::INFINITY;
    }
    payments.iter().zip(deltas).map(|(p, &e)| p * fast_pow(1.0 + rate, -e)).sum()
}

// XNPV first derivative
// \sum_{i=1}^n P_i * (d_0 - d_i) / 365 * (1 + rate)^{((d_0 - d_i)/365 - 1)}}
// simplify in order to reuse cached deltas (d_i - d_0)/365
// \sum_{i=1}^n \frac{P_i * -(d_i - d_0) / 365}{(1 + rate)^{((d_i - d_0)/365 + 1)}}
// fn xnpv_result_deriv(payments: &[f64], deltas: &[f64], rate: f64) -> f64 {
//     payments.iter().zip(deltas).map(|(p, e)| p * -e * fast_pow(1.0 + rate, -e - 1.0)).sum()
// }

fn xnpv_result_with_deriv(payments: &[f64], deltas: &[f64], rate: f64) -> (f64, f64) {
    if rate <= -1.0 {
        return (f64::INFINITY, f64::INFINITY);
    }
    // pow is an expensive function.
    // we can re-use the result of pow for derivative calculation
    payments.iter().zip(deltas).fold((0.0, 0.0), |acc, (p, e)| {
        let y0 = p * fast_pow(1.0 + rate, -e);
        let y1 = y0 * -e / (1.0 + rate);
        (acc.0 + y0, acc.1 + y1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn test_sign_changes() {
        assert_eq!(sign_changes(&[1., 2., 3.]), 0);
        assert_eq!(sign_changes(&[1., 2., -3.]), 1);
        assert_eq!(sign_changes(&[1., -2., 3.]), 2);
        assert_eq!(sign_changes(&[-1., 2., -3.]), 2);
        assert_eq!(sign_changes(&[-1., -2., -3.]), 0);
        assert_eq!(sign_changes(&[1., f64::NAN, 3.]), 0);
    }

    #[rstest]
    fn test_zero_crossing_points() {
        assert_eq!(zero_crossing_points(&[1., 2., 3.]), Vec::<usize>::new());
        assert_eq!(zero_crossing_points(&[1., -2., -3.]), vec![0]);
        assert_eq!(zero_crossing_points(&[1., -2., 3.]), vec![0, 1]);
        assert_eq!(zero_crossing_points(&[-1., -2., 3.]), vec![1]);
        assert_eq!(zero_crossing_points(&[1., f64::NAN, 3.]), Vec::<usize>::new());

        assert_eq!(
            zero_crossing_points(&[7., 6., -3., -4., -7., 8., 3., -6., 7., 8.]),
            vec![1, 4, 6, 7],
        );
    }
}
