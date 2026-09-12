#!/usr/bin/env python3
"""
Module: 3_2_Kinematic_Tracker.py
Kinematic Phase Variance Tracker (KPVT): adaptive background subtraction
(VSS-LMS) feeding a dynamic occupancy detector (OS-CFAR).

The dispatcher's previous kinematic-energy computation subtracted a single
static mean (np.mean(v_abs, axis=0)) from the whole chunk before running PCA,
and compared the result against one hardcoded global KE_THRESHOLD from
config.env. Both pieces are replaced here with adaptive, per-bucket
alternatives that carry state across chunks (see state_store.py): a
per-timestep adaptive background estimate in place of the static mean, and a
per-bucket dynamic detection threshold in place of the single global
constant.
"""
import numpy as np
from scipy.special import gammaln
from sklearn.decomposition import PCA

# VSS-LMS step-size bounds and attack/release rates (see vss_lms_update
# docstring for why attack and release use different rates). mu_max is
# intentionally kept well under the 1-tap stability limit (mu < 2 for a
# constant unit regressor) so a single chunk of motion cannot swing the
# background estimate too far in one step.
LMS_ATTACK = 0.4     # fast: how quickly mu is allowed to DROP when a large residual appears
LMS_RELEASE = 0.97   # slow: how cautiously mu is allowed to RISE back up during quiet
LMS_GAMMA = 1.0
LMS_MU_MIN = 0.01
LMS_MU_MAX = 0.5


def vss_lms_update(bucket_state, sample_magnitude):
    """
    One VSS-LMS step of the per-bucket adaptive background estimate.

    For a scalar (or elementwise-vector) background estimate with a constant
    regressor, the general adaptive-filter update
        w(n+1) = w(n) + mu(n) * x(n) * e*(n)
    collapses to an exponential moving average with an adaptive smoothing
    factor:
        b(n+1) = (1 - mu(n)) * b(n) + mu(n) * y(n)
    This is the practical form implemented below. The step size mu(n) is
    driven toward an instantaneous target derived from the residual power,
    following the inverse of the step-size relationship in Kwong & Johnston,
    "A Variable Step Size LMS Algorithm," IEEE Trans. Signal Processing,
    vol. 40, no. 7, 1992: their rule *raises* mu when the instantaneous error
    is large, which is correct when a filter is converging toward a fixed
    reference (large error = not converged yet). Background subtraction is
    the inverse case -- the "error" here (this sample's magnitude minus the
    current background estimate) is the target's kinematic signal -- so the
    relationship is inverted: target_mu is large when the residual is small
    (background is stable; safe to adapt fast) and near mu_min when the
    residual is large (something is moving; protect the background from
    absorbing it).

    A single symmetric smoothing rate toward that target (as in Kwong &
    Johnston's own recursion) was tried and empirically failed here: with a
    single rate slow enough to hold mu near mu_max through ordinary sensor
    noise during quiet periods, mu could not fall fast enough within one
    short motion burst to stop the background chasing it (validated in the
    module's test run -- mu stayed pinned near mu_max for the whole burst).
    The fix applied is the standard "fast attack / slow release" envelope
    pattern from AGC and audio dynamics processing: mu moves toward a lower
    target quickly (LMS_ATTACK) but is only allowed to climb back toward a
    higher target slowly (LMS_RELEASE), so a burst of motion is caught
    within a sample or two while a brief quiet moment can't immediately
    reopen adaptation and let the next burst leak into the background. This
    attack/release smoothing is this implementation's own addition on top of
    Kwong & Johnston's inverted step-size target, not part of the cited
    paper itself.

    sample_magnitude may be a scalar or an array (one magnitude value per
    V-matrix element); residual_power is always reduced to a single scalar
    driving one shared step size, a deliberate simplification of Kwong &
    Johnston's originally single-channel formulation rather than a
    literature-cited multichannel extension.
    """
    background = bucket_state["lms_background"]

    if background is None:
        bucket_state["lms_background"] = np.array(sample_magnitude, dtype=float)
        bucket_state["lms_mu"] = LMS_MU_MIN  # cautious until a first background estimate exists
        return np.zeros_like(np.array(sample_magnitude, dtype=float))

    mu = bucket_state["lms_mu"]
    residual = np.array(sample_magnitude, dtype=float) - background
    residual_power = float(np.mean(residual ** 2))

    target_mu = LMS_MU_MAX / (1.0 + LMS_GAMMA * residual_power)
    target_mu = min(max(target_mu, LMS_MU_MIN), LMS_MU_MAX)
    rate = LMS_ATTACK if target_mu < mu else LMS_RELEASE
    mu = rate * mu + (1.0 - rate) * target_mu
    mu = min(max(mu, LMS_MU_MIN), LMS_MU_MAX)

    bucket_state["lms_background"] = (1.0 - mu) * background + mu * np.array(sample_magnitude, dtype=float)
    bucket_state["lms_mu"] = mu
    return residual


def _solve_cfar_threshold_factor(n, k, p_fa):
    """
    Bisection solve for the OS-CFAR threshold multiplier T satisfying
    Rohling's exact false-alarm probability equation,
        P_fa(T) = prod_{i=0}^{k-1} (n-i) / (n-i+T)
    evaluated here in log-gamma form for numerical stability at larger n:
        log P_fa(T) = lgamma(n+1) - lgamma(n-k+1) + lgamma(n-k+T+1) - lgamma(n+T+1)
    log P_fa(T) is strictly decreasing in T, so a bounded bisection search
    converges to the unique root.

    Reference: H. Rohling, "Radar CFAR Thresholding in Clutter and Multiple
    Target Situations," IEEE Trans. Aerospace and Electronic Systems,
    AES-19(4), pp. 608-621, 1983.
    """
    log_target = np.log(p_fa)

    def log_pfa(t):
        return (gammaln(n + 1) - gammaln(n - k + 1)
                + gammaln(n - k + t + 1) - gammaln(n + t + 1))

    lo, hi = 1e-6, 1e6
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if log_pfa(mid) > log_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def os_cfar_threshold(reference_cells, k_ratio=0.75, p_fa=1e-3, min_cells=8):
    """
    Order-Statistic CFAR detection threshold (Rohling 1983) over a bucket's
    own rolling kinematic-energy history, replacing a single global
    KE_THRESHOLD with a per-bucket, self-calibrating one: a bucket sitting in
    an intrinsically noisier RF environment (more ambient multipath churn)
    gets a proportionally higher threshold, rather than the same fixed
    constant applied everywhere regardless of local noise floor.

    Sorts the (power-domain) reference cells and takes the k-th order
    statistic (k = floor(k_ratio * N), Rohling's recommended 3N/4) as the
    noise-floor estimate, then scales it by a threshold factor solved for the
    target false-alarm probability p_fa -- rather than an arbitrary
    multiplier -- so the false-alarm rate stays consistent across buckets and
    environments instead of drifting with however "loud" a given
    environment's ambient RF churn happens to be.

    Returns None when there are fewer than min_cells reference samples: OS-
    CFAR needs enough history to characterize the noise floor, and the
    caller is expected to fall back to a fixed default threshold until then.
    """
    n = len(reference_cells)
    if n < min_cells:
        return None

    sorted_cells = np.sort(reference_cells)
    k = min(max(1, int(k_ratio * n)), n - 1)
    noise_estimate = sorted_cells[k - 1]
    threshold_factor = _solve_cfar_threshold_factor(n, k, p_fa)
    return threshold_factor * noise_estimate


def kpvt_module(v_matrices, bucket_state=None):
    """
    Computes the chunk's kinematic-energy scalar. When bucket_state is
    supplied (i.e. STATE_FILE is configured), the background subtracted from
    each timestep is the VSS-LMS adaptive estimate carried over from previous
    chunks rather than this chunk's own static mean -- the improvement this
    module exists to make is specifically visible across chunk boundaries
    and in the sparse-packet regime the static per-chunk mean handled badly
    (too few samples in one chunk to form a stable mean, no memory of the
    chunk before it). Without a bucket_state (e.g. running this file
    standalone, or STATE_FILE disabled), it falls back to the original
    static-mean behavior so the function still degrades gracefully rather
    than failing outright.
    """
    t_steps = v_matrices.shape[0]
    v_abs = np.abs(v_matrices).reshape(t_steps, -1)

    if bucket_state is not None:
        residual = np.stack([vss_lms_update(bucket_state, v_abs[t]) for t in range(t_steps)])
    else:
        residual = v_abs - np.mean(v_abs, axis=0)

    variance_profile = PCA(n_components=1).fit_transform(residual).flatten()
    return float(np.var(variance_profile))
