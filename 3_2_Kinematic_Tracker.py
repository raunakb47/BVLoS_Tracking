#!/usr/bin/env python3
"""
Module: 3_2_Kinematic_Tracker.py
Kinematic Phase Variance Tracker (KPVT): adaptive background subtraction
(VSS-LMS) feeding a dynamic occupancy detector (OS-CFAR).

Both parts are per-bucket and carry state across chunks (state_store.py),
replacing a per-chunk static mean as the background and one global
KE_THRESHOLD as the detection threshold.
"""
import numpy as np
from scipy.special import gammaln
from sklearn.decomposition import PCA

# mu_max stays well under the 1-tap stability limit (mu < 2 for a constant
# unit regressor) so one chunk of motion cannot swing the background far.
LMS_ATTACK = 0.4     # rate at which mu may DROP when the residual rises
LMS_RELEASE = 0.97   # rate at which mu may RISE back during quiet
LMS_GAMMA = 1.0      # applied to the residual/floor ratio, so dimensionless
LMS_MU_MIN = 0.01
LMS_MU_MAX = 0.5

# Residual-power floor that makes the step-size rule scale-free. A low-order
# statistic, not a mean, so a burst inside the window does not pull it up.
LMS_FLOOR_HISTORY = 256
LMS_FLOOR_PERCENTILE = 25.0
LMS_FLOOR_HOLD_RATIO = 4.0   # samples this far above the floor are held out as motion
LMS_FLOOR_FORCE_UPDATE = 512 # ... for at most this many, so a real level change is re-learned
LMS_FLOOR_MIN_CELLS = 8      # below this, accept everything so the floor can bootstrap


def _residual_power_floor(bucket_state, residual_power):
    """
    Track and return the bucket's quiet-time residual power, the divisor in
    the step-size rule.

    Samples far above the floor are withheld from the history, otherwise
    sustained motion raises the floor and re-opens adaptation on itself. The
    withholding is bounded so a real level change (AGC step, different
    beamformee grouping) is eventually absorbed. Plain list, not a deque: the
    bucket state round-trips through np.save.
    """
    history = bucket_state.setdefault("lms_floor_history", [])
    held = bucket_state.setdefault("lms_floor_hold", 0)

    if len(history) < LMS_FLOOR_MIN_CELLS:
        history.append(residual_power)
    else:
        floor = float(np.percentile(history, LMS_FLOOR_PERCENTILE))
        if residual_power <= LMS_FLOOR_HOLD_RATIO * floor or held >= LMS_FLOOR_FORCE_UPDATE:
            history.append(residual_power)
            bucket_state["lms_floor_hold"] = 0
        else:
            bucket_state["lms_floor_hold"] = held + 1

    if len(history) > LMS_FLOOR_HISTORY:
        del history[:-LMS_FLOOR_HISTORY]

    return float(np.percentile(history, LMS_FLOOR_PERCENTILE))


def vss_lms_update(bucket_state, sample_magnitude):
    """
    One VSS-LMS step of the per-bucket adaptive background estimate.
    sample_magnitude is a scalar or one magnitude per V-matrix element.

    With a constant regressor the LMS update w(n+1) = w(n) + mu(n) x(n) e*(n)
    collapses to an EMA with an adaptive smoothing factor,
        b(n+1) = (1 - mu(n)) b(n) + mu(n) y(n),
    which is the form used here. mu is driven toward a target derived from the
    residual power, inverting the step-size relationship of Kwong & Johnston,
    IEEE Trans. Signal Processing 40(7), 1992: their rule raises mu on a large
    error, correct when converging toward a fixed reference. Here the residual
    is the target's kinematic signal, not a convergence error, so the sense is
    flipped -- mu is large while the residual sits at the bucket's noise floor
    and near mu_min once it rises above it.

    The ratio fed to that rule is residual_power / floor, not residual_power.
    An absolute-scale rule does not work on unit-normalized V-matrices:
    measured per-sample residual power runs ~5e-2, where LMS_MU_MAX / (1 +
    residual_power) is still ~95% of mu_max, so mu never leaves its ceiling and
    the background becomes a 2-sample EMA that tracks the target instead of the
    environment. Still-vs-moving contrast measured 67x that way against 2735x
    for a static per-chunk mean; dividing by the floor gives 2285x.

    A single symmetric rate toward the target (as in the cited recursion) also
    fails: a rate slow enough to hold mu near mu_max through sensor noise
    cannot drop it inside one short burst. mu therefore attacks fast and
    releases slowly. The envelope and the floor normalization are both
    additions on top of the cited step-size target.

    Limitation: any cross-chunk adaptive background reads a step change in
    level as one chunk of motion, where a static per-chunk mean is immune by
    construction. Floor normalization improves motion-vs-level-shift
    discrimination ~4x but does not remove it.

    residual_power is reduced to one scalar driving a single shared step size,
    a simplification of the cited single-channel formulation rather than a
    published multichannel extension.
    """
    background = bucket_state["lms_background"]

    if background is None:
        bucket_state["lms_background"] = np.array(sample_magnitude, dtype=float)
        bucket_state["lms_mu"] = LMS_MU_MIN  # cautious until a first background estimate exists
        return np.zeros_like(np.array(sample_magnitude, dtype=float))

    mu = bucket_state["lms_mu"]
    residual = np.array(sample_magnitude, dtype=float) - background
    residual_power = float(np.mean(residual ** 2))

    floor = _residual_power_floor(bucket_state, residual_power)
    normalized_power = residual_power / max(floor, 1e-12)

    target_mu = LMS_MU_MAX / (1.0 + LMS_GAMMA * normalized_power)
    target_mu = min(max(target_mu, LMS_MU_MIN), LMS_MU_MAX)
    rate = LMS_ATTACK if target_mu < mu else LMS_RELEASE
    mu = rate * mu + (1.0 - rate) * target_mu
    mu = min(max(mu, LMS_MU_MIN), LMS_MU_MAX)

    bucket_state["lms_background"] = (1.0 - mu) * background + mu * np.array(sample_magnitude, dtype=float)
    bucket_state["lms_mu"] = mu
    return residual


def _solve_cfar_threshold_factor(n, k, p_fa):
    """
    Solve Rohling's exact OS-CFAR false-alarm equation for the threshold
    multiplier T. Rohling, IEEE Trans. AES 19(4):608-621, 1983.

        P_fa(T) = prod_{i=0}^{k-1} (n-i) / (n-i+T)

    Evaluated in log-gamma form for stability at larger n. log P_fa is
    strictly decreasing in T, so bisection converges to the unique root.
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
    Order-statistic CFAR threshold (Rohling 1983) over a bucket's own rolling
    kinematic-energy history, so a bucket in a noisier RF environment gets a
    proportionally higher threshold than one global constant would give it.

    Takes the k-th order statistic of the reference cells (k = 3N/4, Rohling's
    recommendation) as the noise floor and scales it by the factor solved for
    p_fa, which keeps the false-alarm rate comparable across buckets.

    Returns None below min_cells reference samples; the caller falls back to a
    fixed threshold until then.
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
    Return the chunk's kinematic-energy scalar.

    With bucket_state (STATE_FILE configured), the background subtracted at
    each timestep is the VSS-LMS estimate carried over from previous chunks,
    which is what helps in the sparse regime where one chunk holds too few
    samples for a stable mean. Without it, falls back to this chunk's static
    mean so the function still runs standalone.
    """
    t_steps = v_matrices.shape[0]
    v_abs = np.abs(v_matrices).reshape(t_steps, -1)

    if bucket_state is not None:
        residual = np.stack([vss_lms_update(bucket_state, v_abs[t]) for t in range(t_steps)])
    else:
        residual = v_abs - np.mean(v_abs, axis=0)

    variance_profile = PCA(n_components=1).fit_transform(residual).flatten()
    return float(np.var(variance_profile))
