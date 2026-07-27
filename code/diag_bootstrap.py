"""
diag_bootstrap.py
=================
Is the plug-in weight trustworthy at the sample sizes of interest?

The weight depends on v_S, v_N and their covariance, all estimated by a
within-season bootstrap.  Bootstrap variance estimates for EXTREME sample
quantiles are known to be poor in small samples, and the weight is built
from a difference of two such estimates, so any downward bias in the
estimated variance of the difference inflates the estimated squared
discrepancy and therefore the weight.

This script compares, by Monte Carlo over many independent buffers:

    truth       Var and Cov of (theta_S, theta_N) across buffers, and the
                true discrepancy Delta = E[theta_N] - theta_true
    bootstrap   the mean of the within-buffer bootstrap estimates
    weights     the true MSE-optimal lambda against the mean plug-in one
"""

import numpy as np
import shrinkage as sh

S = 12
LEVELS = np.array([0.05, 0.95])
N_MC = 1500
N_BOOT = 200


def buffers_scale(Ts, rng, sigma_high=2.0, high=(1, 2, 8, 9)):
    return {s: (sigma_high if s in high else 1.0)
            * rng.standard_t(3, size=Ts) / np.sqrt(3.0)
            for s in range(1, S + 1)}


def buffers_shape(Ts, rng, skew=(1, 2, 8, 9)):
    out = {}
    for s in range(1, S + 1):
        if s in skew:
            z = np.exp(rng.standard_normal(Ts))
            out[s] = (z - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
        else:
            out[s] = rng.standard_normal(Ts)
    return out


def truth_quantiles(kind, s_star, levels, rng, n=400000):
    """Population quantile of season s_star, by brute force."""
    if kind == "scale":
        sig = 2.0 if s_star in (1, 2, 8, 9) else 1.0
        x = sig * rng.standard_t(3, size=n) / np.sqrt(3.0)
    else:
        if s_star in (1, 2, 8, 9):
            z = np.exp(rng.standard_normal(n))
            x = (z - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
        else:
            x = rng.standard_normal(n)
    return np.quantile(x, levels)


def run(kind, make, Ts, s_star=1):
    rng = np.random.default_rng(11)
    theta_true = truth_quantiles(kind, s_star, LEVELS, rng)

    S_mc = np.empty((N_MC, LEVELS.size))
    N_mc = np.empty((N_MC, LEVELS.size))
    boot = {k: np.zeros(LEVELS.size) for k in ("v_S", "v_N", "cov")}
    lam_hat = np.zeros(LEVELS.size)

    for i in range(N_MC):
        buf = make(Ts, rng)
        tS, tN = sh.scheme_quantiles(buf, s_star, LEVELS)
        S_mc[i], N_mc[i] = tS, tN
        if i < 300:          # bootstrap is expensive; a subsample suffices
            m = sh.bootstrap_moments(buf, s_star, LEVELS, N_BOOT, rng)
            for k in boot:
                boot[k] += m[k] / 300
            lam_hat += sh.lambda_plugin(buf, s_star, LEVELS,
                                        N_BOOT, rng) / 300

    v_S = S_mc.var(axis=0, ddof=1)
    v_N = N_mc.var(axis=0, ddof=1)
    cov = np.array([np.cov(S_mc[:, j], N_mc[:, j], ddof=1)[0, 1]
                    for j in range(LEVELS.size)])
    delta = N_mc.mean(axis=0) - theta_true
    bias_S = S_mc.mean(axis=0) - theta_true

    lam_star = np.clip((delta ** 2 + v_N - cov)
                       / np.maximum(delta ** 2 + v_N + v_S - 2 * cov, 1e-12),
                       0, 1)

    print(f"\n=== {kind} DGP, Ts = {Ts}, season {s_star} ===")
    for j, p in enumerate(LEVELS):
        print(f"  p = {p:.2f}")
        print(f"    true      v_S={v_S[j]:.4f}  v_N={v_N[j]:.4f}  "
              f"cov={cov[j]:.4f}  Var(diff)={v_S[j]+v_N[j]-2*cov[j]:.4f}")
        print(f"    bootstrap v_S={boot['v_S'][j]:.4f}  v_N={boot['v_N'][j]:.4f}  "
              f"cov={boot['cov'][j]:.4f}  Var(diff)="
              f"{boot['v_S'][j]+boot['v_N'][j]-2*boot['cov'][j]:.4f}")
        print(f"    Delta={delta[j]:+.4f}  bias(theta_S)={bias_S[j]:+.4f}  "
              f"lambda*={lam_star[j]:.3f}  mean lambda_hat={lam_hat[j]:.3f}")


if __name__ == "__main__":
    for Ts in (20, 50):
        run("scale", buffers_scale, Ts)
        run("shape", buffers_shape, Ts)
