"""
test_shrinkage.py
=================
Unit checks for the shrinkage module, run on residual buffers drawn
directly from known distributions, so that the behaviour of the weight is
isolated from the forecasting pipeline.

What the weight should do:
  pure scale heterogeneity  -> lambda near 0 (normalisation is correct
                               and more efficient)
  shape heterogeneity       -> lambda near 1 for the odd seasons
  larger Ts                 -> lambda closer to 1 under shape, since the
                               stratified estimator gets cheaper
"""

import numpy as np
import shrinkage as sh

S = 12
ALPHA = 0.10
LEVELS = np.array([0.05, 0.95])


def buffers_scale(Ts, rng, sigma_high=2.0, high=(1, 2, 8, 9)):
    """Common shape (t3, standardised), season-specific scale."""
    out = {}
    for s in range(1, S + 1):
        sig = sigma_high if s in high else 1.0
        out[s] = sig * rng.standard_t(3, size=Ts) / np.sqrt(3.0)
    return out


def buffers_shape(Ts, rng, skew=(1, 2, 8, 9)):
    """Common unit scale, four seasons right-skewed (standardised lognormal)."""
    out = {}
    for s in range(1, S + 1):
        if s in skew:
            z = np.exp(rng.standard_normal(Ts))
            out[s] = (z - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
        else:
            out[s] = rng.standard_normal(Ts)
    return out


def mean_lambda(make, Ts, n_draw=40, seed=0, seasons=(1, 5)):
    rng = np.random.default_rng(seed)
    acc = {s: [] for s in seasons}
    for _ in range(n_draw):
        buf = make(Ts, rng)
        for s in seasons:
            lam = sh.lambda_plugin(buf, s, LEVELS, n_boot=120, rng=rng)
            acc[s].append(lam.mean())
    return {s: float(np.mean(v)) for s, v in acc.items()}


if __name__ == "__main__":
    print("mean plug-in lambda (season 1 = odd season, season 5 = ordinary)\n")
    print(f"{'DGP':<10} {'Ts':>4} {'season 1':>10} {'season 5':>10}")
    for Ts in [10, 20, 35, 50, 100]:
        r = mean_lambda(buffers_scale, Ts, seed=1)
        print(f"{'scale':<10} {Ts:>4} {r[1]:>10.3f} {r[5]:>10.3f}")
    print()
    for Ts in [10, 20, 35, 50, 100]:
        r = mean_lambda(buffers_shape, Ts, seed=2)
        print(f"{'shape':<10} {Ts:>4} {r[1]:>10.3f} {r[5]:>10.3f}")

    # A direct check of the algebra: with Delta known to be zero the weight
    # should track v_N / (v_N + v_S) up to the covariance correction.
    print("\nsanity: Delta = 0 by construction (all seasons identical)")
    rng = np.random.default_rng(7)
    for Ts in [20, 50]:
        buf = {s: rng.standard_normal(Ts) for s in range(1, S + 1)}
        m = sh.bootstrap_moments(buf, 1, LEVELS, n_boot=400, rng=rng)
        lam = sh.lambda_plugin(buf, 1, LEVELS, n_boot=400, rng=rng)
        ratio = m["v_N"] / (m["v_N"] + m["v_S"])
        print(f"  Ts={Ts:>3}  lambda={np.round(lam,3)}  "
              f"v_N/(v_N+v_S)={np.round(ratio,3)}  "
              f"v_S={np.round(m['v_S'],4)}  v_N={np.round(m['v_N'],4)}  "
              f"cov={np.round(m['cov'],4)}")
