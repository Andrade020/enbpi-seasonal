"""
fastcal.py
==========
Fast primitives for the seasonal calibration family.

Two ideas make the weight grid cheap.  First, both endpoint estimators are
evaluated once per buffer at every level the line search needs, so the
weight grid costs nothing but arithmetic.  Second, the evaluation sample is
sorted once and equipped with prefix sums, so the exact mean coverage and
mean Winkler score of an interval are obtained in O(log n) instead of a
pass over the sample.

A note on the finite-sample conformal endpoints.  The conformal
prescription is an ORDER STATISTIC, not a quantile level: the upper
endpoint is the k-th smallest residual with k = ceil((n+1)(1-alpha/2)),
and the lower endpoint is the k-th smallest with k = floor((n+1)alpha/2).
Feeding k/n to an interpolating quantile function is not the same thing
and it undoes most of the correction at the lower end, where k is 1 or 2:
with level k/n the interpolation returns the k-th order statistic plus
(1 - k/n) of the gap to the next one, which at k = 1 is almost the whole
gap.  The functions below therefore index order statistics directly.

Two sample sizes matter, and they are different:

    n >= 2/alpha - 1     both conformal indices exist, so a finite
                         interval exists at all (19 at alpha = 0.10)
    n >= 4/alpha - 1     the indices fall strictly inside the sample, so
                         the endpoints are not the buffer extremes
                         (39 at alpha = 0.10)

Between the two the conformal interval IS the range of the buffer.  Below
the first, the prescription is unbounded; `conformal_indices` reports that
case instead of hiding it.
"""

import numpy as np

MAD_CONST = 0.6744897501960817


def mad_scale(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 1.0
    s = float(np.median(np.abs(x - np.median(x))) / MAD_CONST)
    if not np.isfinite(s) or s <= 1e-8:
        s = float(np.std(x))
    return s if (np.isfinite(s) and s > 1e-8) else 1.0


def q_sorted(a, ps):
    """numpy 'linear' quantiles from an already sorted array."""
    n = len(a)
    ps = np.asarray(ps, dtype=float)
    if n == 1:
        return np.full(ps.shape, a[0], dtype=float)
    h = (n - 1) * np.clip(ps, 0.0, 1.0)
    lo = np.floor(h).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    return a[lo] + (h - lo) * (a[hi] - a[lo])


def conformal_indices(p_lo, p_hi, n):
    """
    One-based order-statistic indices of the conformal endpoints, together
    with a flag marking the levels at which the prescription is unbounded
    (lower index below 1 or upper index above n).  Indices are clipped so
    that the caller still receives usable endpoints, namely the buffer
    extremes, but the flag says the guarantee does not hold there.
    """
    p_lo = np.atleast_1d(np.asarray(p_lo, dtype=float))
    p_hi = np.atleast_1d(np.asarray(p_hi, dtype=float))
    k_lo = np.floor((n + 1) * p_lo).astype(int)
    k_hi = np.ceil((n + 1) * p_hi).astype(int)
    unbounded = (k_lo < 1) | (k_hi > n)
    return np.clip(k_lo, 1, n), np.clip(k_hi, 1, n), unbounded


def order_stat(a_sorted, k):
    """k is one-based."""
    return a_sorted[np.clip(np.asarray(k, dtype=int), 1, len(a_sorted)) - 1]


def season_scales(res_by_season):
    """Robust scale of every season, computed once and reused."""
    return {s: mad_scale(v) for s, v in res_by_season.items()}


class Endpoints:
    """
    Both schemes evaluated on one residual buffer, over the whole beta grid.

    Attributes S_lo, S_hi, N_lo, N_hi have the shape of the beta grid, so
    the interval for any weight is a convex combination followed by an
    argmin over beta.

    corrected=True uses the conformal order statistics; n_levels_S allows
    the conformal indices of the stratified endpoint to be computed for a
    buffer size other than the one at hand, which is what cross-validation
    needs so that the levels it scores are the levels that will be used at
    prediction time.
    """

    def __init__(self, res_by_season, s_star, alpha, beta_grid,
                 corrected=False, n_levels_S=None, scales=None):
        own = np.sort(np.asarray(res_by_season[s_star], dtype=float))
        if scales is None:
            scales = {s: mad_scale(v) for s, v in res_by_season.items()}
        z = np.sort(np.concatenate(
            [np.asarray(v, dtype=float) / scales[s]
             for s, v in res_by_season.items()]))
        Ts, T = len(own), len(z)
        sc = scales[s_star]

        lo_lv = np.atleast_1d(beta_grid).astype(float)
        hi_lv = 1.0 - alpha + lo_lv

        if corrected:
            nS = Ts if n_levels_S is None else n_levels_S
            kS_lo, kS_hi, unb_S = conformal_indices(lo_lv, hi_lv, nS)
            kN_lo, kN_hi, unb_N = conformal_indices(lo_lv, hi_lv, T)
            self.S_lo = order_stat(own, kS_lo)
            self.S_hi = order_stat(own, kS_hi)
            self.N_lo = sc * order_stat(z, kN_lo)
            self.N_hi = sc * order_stat(z, kN_hi)
            self.unbounded = unb_S | unb_N
        else:
            self.S_lo = q_sorted(own, lo_lv)
            self.S_hi = q_sorted(own, hi_lv)
            self.N_lo = sc * q_sorted(z, lo_lv)
            self.N_hi = sc * q_sorted(z, hi_lv)
            self.unbounded = np.zeros(len(lo_lv), dtype=bool)

    def interval(self, lam):
        """Offsets of the hybrid interval for weight lam."""
        lo_all = lam * self.S_lo + (1 - lam) * self.N_lo
        hi_all = lam * self.S_hi + (1 - lam) * self.N_hi
        b = int(np.argmin(hi_all - lo_all))
        return float(lo_all[b]), float(hi_all[b])

    def any_unbounded(self):
        return bool(self.unbounded.any())


class Evaluator:
    """
    Exact mean coverage and mean Winkler score of an interval against a
    fixed evaluation sample, in O(log n) per interval.

    For a sorted sample y with prefix sums P,
        E[(lo - y)_+] = lo * F(lo) - P(lo),
        E[(y - hi)_+] = (P(inf) - P(hi)) - hi * (1 - F(hi)).
    """

    def __init__(self, y, alpha):
        self.y = np.sort(np.asarray(y, dtype=float))
        self.n = len(self.y)
        self.cum = np.concatenate([[0.0], np.cumsum(self.y)])
        self.alpha = alpha

    def _below(self, x):
        k = int(np.searchsorted(self.y, x, side="right"))
        return k / self.n, self.cum[k] / self.n

    def coverage(self, lo, hi):
        klo = np.searchsorted(self.y, lo, side="left")
        khi = np.searchsorted(self.y, hi, side="right")
        return float((khi - klo) / self.n)

    def score(self, lo, hi):
        Flo, Plo = self._below(lo)
        Fhi, Phi = self._below(hi)
        tot = self.cum[-1] / self.n
        left = lo * Flo - Plo
        right = (tot - Phi) - hi * (1.0 - Fhi)
        return float((hi - lo) + (2.0 / self.alpha) * (left + right))


def winkler(lo, hi, y, alpha):
    """Winkler interval score of single or vector outcomes."""
    y = np.asarray(y, dtype=float)
    s = (hi - lo) + (2.0 / alpha) * (
        np.where(y < lo, lo - y, 0.0) + np.where(y > hi, y - hi, 0.0))
    return s
