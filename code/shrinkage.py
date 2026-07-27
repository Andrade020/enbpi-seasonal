"""
shrinkage.py
============
The one-parameter family of seasonal calibration schemes and the rules
that pick its weight from data.

For a season s and a quantile level p, two estimators of the
season-specific residual quantile are available from the same buffer:

    theta_S     empirical p-quantile of the Ts residuals of season s
    theta_N     sigma_hat_s times the p-quantile of the pooled
                standardised residuals eps_t / sigma_hat_{s(t)}

and the family is their convex combination

    theta(lam) = lam * theta_S + (1 - lam) * theta_N .

lam = 1 is the stratified scheme (EnbPI-S), lam = 0 the normalised one
(EnbPI-N).  Three ways of choosing lam are implemented:

    plugin    positive-part James-Stein weight built from a within-season
              bootstrap of (theta_S, theta_N)
    cv        K-fold cross-validation over the season buffer, minimising
              the Winkler interval score
    oracle    the weight that minimises the realised interval score on the
              evaluation sample; not feasible in practice, used only to
              bound what a selection rule can achieve

Everything here operates on residual buffers, so it is independent of the
forecasting model that produced them.
"""

import numpy as np

MAD_CONST = 0.6744897501960817     # Phi^{-1}(0.75)


# ---------------------------------------------------------------- basics
def mad_scale(x):
    """Robust scale: MAD / Phi^{-1}(0.75), with guards for degenerate input."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 1.0
    s = float(np.median(np.abs(x - np.median(x))) / MAD_CONST)
    if not np.isfinite(s) or s <= 1e-8:
        s = float(np.std(x))
    return s if (np.isfinite(s) and s > 1e-8) else 1.0


def q_sorted(a_sorted, ps):
    """Linear-interpolation quantiles of a sorted sample (numpy 'linear')."""
    n = len(a_sorted)
    ps = np.atleast_1d(np.asarray(ps, dtype=float))
    if n == 1:
        return np.full(ps.shape, a_sorted[0], dtype=float)
    h = (n - 1) * np.clip(ps, 0.0, 1.0)
    lo = np.floor(h).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    return a_sorted[lo] + (h - lo) * (a_sorted[hi] - a_sorted[lo])


def winkler_score(lo, hi, y, alpha):
    """Interval score: width plus 2/alpha times the miss distance."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    y = np.asarray(y, dtype=float)
    s = hi - lo
    s = s + (2.0 / alpha) * np.where(y < lo, lo - y, 0.0)
    s = s + (2.0 / alpha) * np.where(y > hi, y - hi, 0.0)
    return s


# ------------------------------------------------------- the two schemes
def scheme_quantiles(res_by_season, s_star, levels):
    """
    Both estimators at the requested quantile levels.

    res_by_season : dict season -> 1d array of residuals
    returns (theta_S, theta_N) each of shape levels.shape
    """
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    own = np.sort(np.asarray(res_by_season[s_star], dtype=float))
    theta_S = q_sorted(own, levels)

    scales = {s: mad_scale(v) for s, v in res_by_season.items()}
    z = np.concatenate([np.asarray(v, dtype=float) / scales[s]
                        for s, v in res_by_season.items()])
    theta_N = scales[s_star] * q_sorted(np.sort(z), levels)
    return theta_S, theta_N


def bootstrap_moments(res_by_season, s_star, levels, n_boot=200, rng=None):
    """
    Within-season nonparametric bootstrap of (theta_S, theta_N).

    Every season is resampled to its own size, so the season scales are
    re-estimated on each replicate and their uncertainty enters v_N and the
    covariance rather than being ignored.

    Returns dict with v_S, v_N, cov, each an array over `levels`.
    """
    rng = np.random.default_rng() if rng is None else rng
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    S_draws = np.empty((n_boot, levels.size))
    N_draws = np.empty((n_boot, levels.size))
    seasons = list(res_by_season.keys())
    arrays = {s: np.asarray(res_by_season[s], dtype=float) for s in seasons}

    for b in range(n_boot):
        boot = {s: arrays[s][rng.integers(0, len(arrays[s]), len(arrays[s]))]
                for s in seasons}
        tS, tN = scheme_quantiles(boot, s_star, levels)
        S_draws[b] = tS
        N_draws[b] = tN

    v_S = S_draws.var(axis=0, ddof=1)
    v_N = N_draws.var(axis=0, ddof=1)
    cov = np.array([np.cov(S_draws[:, j], N_draws[:, j], ddof=1)[0, 1]
                    for j in range(levels.size)])
    return {"v_S": v_S, "v_N": v_N, "cov": cov}


# ------------------------------------------------------- weight: plug-in
def lambda_plugin(res_by_season, s_star, levels, n_boot=200, rng=None,
                  pool_endpoints=False):
    """
    Positive-part plug-in weight,

        lam = (D2 + v_N - c) / (D2 + v_N + v_S - 2c),
        D2  = max(0, (theta_S - theta_N)^2 - (v_S + v_N - 2c)),

    clipped to [0, 1].  With pool_endpoints=True the squared discrepancy is
    averaged over the requested levels before the weight is formed, which
    stabilises it when the buffers are short.
    """
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    tS, tN = scheme_quantiles(res_by_season, s_star, levels)
    m = bootstrap_moments(res_by_season, s_star, levels, n_boot, rng)
    v_S, v_N, c = m["v_S"], m["v_N"], m["cov"]

    var_diff = np.maximum(v_S + v_N - 2 * c, 1e-12)
    D2 = np.maximum(0.0, (tS - tN) ** 2 - var_diff)
    if pool_endpoints:
        D2 = np.full_like(D2, D2.mean())

    lam = (D2 + v_N - c) / np.maximum(D2 + v_N + v_S - 2 * c, 1e-12)
    return np.clip(lam, 0.0, 1.0)


# ------------------------------------------------------------ weight: CV
def lambda_cv(res_by_season, s_star, alpha, lam_grid, n_folds=5, rng=None,
              beta_grid=None):
    """
    K-fold cross-validation over the season-s* buffer.

    Each fold is held out, both schemes are re-estimated on the remaining
    residuals (the other seasons are always used in full, as they would be
    at prediction time), the interval implied by each candidate lam is
    formed, and the Winkler score of the held-out residuals is accumulated.
    Returns the minimising lam.
    """
    rng = np.random.default_rng() if rng is None else rng
    own = np.asarray(res_by_season[s_star], dtype=float)
    n = len(own)
    if n < 2 * n_folds:
        n_folds = max(2, n // 2)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    if beta_grid is None:
        beta_grid = np.linspace(0.0, alpha, 21)

    total = np.zeros(len(lam_grid))
    for f in folds:
        keep = np.setdiff1d(np.arange(n), f, assume_unique=False)
        if len(keep) < 2:
            continue
        sub = dict(res_by_season)
        sub[s_star] = own[keep]
        levels = np.concatenate([beta_grid, 1.0 - alpha + beta_grid])
        tS, tN = scheme_quantiles(sub, s_star, levels)
        k = len(beta_grid)
        for j, lam in enumerate(lam_grid):
            q = lam * tS + (1 - lam) * tN
            lo_all, hi_all = q[:k], q[k:]
            b = int(np.argmin(hi_all - lo_all))
            total[j] += winkler_score(lo_all[b], hi_all[b],
                                      own[f], alpha).sum()
    return float(lam_grid[int(np.argmin(total))])


# ------------------------------------------ interval from a chosen weight
def hybrid_interval(res_by_season, s_star, alpha, lam_lo, lam_hi,
                    beta_grid=None, coherent=True):
    """
    Offsets (lo, hi) of the hybrid interval.

    coherent=True runs a single line search on the shrunken quantile
    function: for every candidate beta both schemes are evaluated at beta
    and at 1-alpha+beta, the endpoints are shrunk, and the beta minimising
    the width of the shrunken interval is chosen.

    coherent=False reproduces the simpler construction of Experiment E8:
    each scheme runs its own line search and only the two resulting
    endpoints are combined.

    lam_lo and lam_hi are the weights for the lower and the upper endpoint.
    """
    if beta_grid is None:
        beta_grid = np.linspace(0.0, alpha, 51)
    k = len(beta_grid)
    levels = np.concatenate([beta_grid, 1.0 - alpha + beta_grid])
    tS, tN = scheme_quantiles(res_by_season, s_star, levels)
    S_lo, S_hi = tS[:k], tS[k:]
    N_lo, N_hi = tN[:k], tN[k:]

    if coherent:
        lo_all = lam_lo * S_lo + (1 - lam_lo) * N_lo
        hi_all = lam_hi * S_hi + (1 - lam_hi) * N_hi
        b = int(np.argmin(hi_all - lo_all))
        return float(lo_all[b]), float(hi_all[b])

    bS = int(np.argmin(S_hi - S_lo))
    bN = int(np.argmin(N_hi - N_lo))
    lo = lam_lo * S_lo[bS] + (1 - lam_lo) * N_lo[bN]
    hi = lam_hi * S_hi[bS] + (1 - lam_hi) * N_hi[bN]
    return float(lo), float(hi)
