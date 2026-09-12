#!/usr/bin/env python3
"""
Module: 3_1_Spatial_Algorithms.py
Spatial Subspace Extractor (SSE) algorithms.

CA-ESPRIT, SpotFi and Residual 2D-MUSIC all previously hardcoded a rank-1
signal-subspace assumption (top eigenvector = the entire signal, everything
else = noise), regardless of what the actual eigenvalue spectrum looked like.
That assumption is only valid when exactly one propagation path dominates the
observation; through-wall/indoor multipath -- the framework's own target
scenario -- routinely violates it, and a rank-1 estimate computed on a
multi-path-contaminated covariance matrix reports whichever path happens to
be instantaneously strongest, not necessarily the direct line-of-sight path.

_signal_subspace_confidence() below estimates whether that assumption was
actually justified for a given covariance matrix, using the array's own
eigenvalue spectrum, and every eigen-decomposition-based algorithm now
returns (ap_aod, confidence) instead of a bare angle. Stage 3 uses the
confidence flag to decide whether this chunk's angle is trustworthy enough to
update a client's tracked position, or whether the estimate should be
withheld this chunk (deferring to KPVT's occupancy signal and the previous
confirmed position) -- i.e. KPVT and SSE operate as complementary signals
rather than a strict "SSE always wins when it runs" pipeline.
"""
import os

import numpy as np
from scipy import linalg

# Wax-Kailath MDL source-count test requires two leftover "noise-only"
# eigenvalues to compare (see _signal_subspace_confidence docstring); below
# this many antenna elements the test is structurally degenerate, and a
# plain eigenvalue-ratio heuristic is used instead.
_MDL_MIN_ELEMENTS = 3

# Minimum ratio between the strongest and second-strongest eigenvalue for a
# single dominant path to be considered established. This is a pragmatic
# threshold (roughly a 5x/7dB power gap), not a literature-derived constant --
# unlike the MDL test, there is no standard closed-form P_fa for it, so treat
# it as a tunable gate rather than a statistically calibrated one.
#
# Both thresholds are env-overridable because neither is calibrated against
# ground-truth-labelled captures yet, and measurements on the bundled Wi-BFI
# traces show them disagreeing on real data: an 11ac 3x1 capture (M=3) gave
# eigenvalue ratios of 1.6-1.8 (below this threshold) while MDL reported a
# single source, and an 11ax 4x2 capture (M=4) gave ratios of 5.9-15.4 (well
# above) while MDL reported two. Requiring both to pass therefore rejected
# every real chunk tested. Until these are calibrated the operator needs to be
# able to move them without editing code; see the README's open-issues note.
_EIGENVALUE_DOMINANCE_RATIO = float(os.getenv("SSE_EIGENVALUE_DOMINANCE_RATIO", 3.0))

# Largest MDL source count still treated as "rank-1 is a usable approximation".
# Raising it to 2 accepts a weak second path rather than withholding the angle.
_MDL_MAX_SOURCES = int(os.getenv("SSE_MDL_MAX_SOURCES", 1))


def _mdl_source_count(eigenvalues_desc, n_snapshots):
    """
    Wax & Kailath, "Detection of Signals by Information Theoretic Criteria,"
    IEEE Trans. ASSP, vol. 33, no. 2, pp. 387-392, 1985 -- the standard
    eigenvalue-based source-count estimator for narrowband array processing.

    For an M-element array and a hypothesis of k signal sources, the (M-k)
    smallest eigenvalues should all equal the noise floor; MDL(k) scores how
    close their geometric/arithmetic mean ratio is to 1 (a perfectly flat
    noise floor) against a complexity penalty that grows with k, and the
    number of sources is estimated as argmin_k MDL(k).

    The test is only informative while at least 2 eigenvalues remain in the
    "noise" hypothesis (M - k >= 2): with exactly one remaining eigenvalue,
    its geometric and arithmetic mean are trivially identical, so that
    hypothesis always scores a perfect (and meaningless) fit. k is therefore
    only searched over the range where M - k >= 2.
    """
    m = len(eigenvalues_desc)
    best_k, best_mdl = 0, np.inf
    for k in range(0, m - 1):  # k = 0 .. m-2, keeping M-k >= 2 noise eigenvalues
        remaining = eigenvalues_desc[k:]
        remaining = np.clip(remaining, 1e-15, None)  # guard log(0) on degenerate/synthetic input
        geometric_mean = np.exp(np.mean(np.log(remaining)))
        arithmetic_mean = np.mean(remaining)
        log_likelihood = (m - k) * n_snapshots * np.log(arithmetic_mean / geometric_mean)
        penalty = 0.5 * k * (2 * m - k) * np.log(n_snapshots)
        mdl = log_likelihood + penalty
        if mdl < best_mdl:
            best_k, best_mdl = k, mdl
    return best_k


def _signal_subspace_confidence(eigenvalues_asc, n_snapshots):
    """
    Decide whether a rank-1 signal-subspace split is justified for this
    covariance matrix. Returns "HIGH" when the data looks like a single
    dominant path (rank-1 valid, matching the algorithms' existing math) and
    "LOW" when a second comparably-strong path is present (rank-1 would blend
    two paths together and the resulting angle should not be trusted as a
    position update).

    Below _MDL_MIN_ELEMENTS antennas, Wax-Kailath MDL cannot distinguish "1
    path" from "2+ paths" at all (see _mdl_source_count) -- a 2-element array
    (this framework's most common configuration) only ever has one testable
    hypothesis under MDL, which is a real ceiling on what any subspace method
    can resolve on that little aperture, not a bug in this implementation.
    The eigenvalue-ratio check is applied regardless of array size as a
    cruder but always-available fallback signal.
    """
    eigenvalues_desc = np.sort(eigenvalues_asc)[::-1]
    dominance_ratio = eigenvalues_desc[0] / max(eigenvalues_desc[1], 1e-15)
    ratio_says_single_path = dominance_ratio >= _EIGENVALUE_DOMINANCE_RATIO

    if len(eigenvalues_desc) < _MDL_MIN_ELEMENTS:
        return "HIGH" if ratio_says_single_path else "LOW"

    mdl_source_count = _mdl_source_count(eigenvalues_desc, n_snapshots)
    return "HIGH" if (mdl_source_count <= _MDL_MAX_SOURCES and ratio_says_single_path) else "LOW"


def algo_ca_esprit(v_matrices, nt):
    """ CA-ESPRIT Analytical solver. Conjugate augmented for low-aperture. """
    v_avg = np.mean(v_matrices, axis=0)
    z = v_avg[:, :, 0].T

    if nt == 2:
        z_conj = np.dot(np.array([[0, 1], [1, 0]]), np.conj(z))
        y = np.vstack([z, z_conj])
    else: y = z

    n_snapshots = v_matrices.shape[1]
    R = np.dot(y, np.conj(y).T) / n_snapshots
    vals, vecs = linalg.eigh(R)
    u_s = vecs[:, -1:]
    confidence = _signal_subspace_confidence(vals, n_snapshots)

    phi = np.dot(linalg.pinv(u_s[:-1, :]), u_s[1:, :])
    ap_aod = np.degrees(np.arcsin(np.clip(np.angle(linalg.eigvals(phi)[0]) / np.pi, -1, 1)))
    return float(ap_aod), confidence

def algo_spotfi(v_matrices, nt):
    """ Spot-Fi Forward-Backward smoothing. Decorrelates indoor coherent multipath. """
    v_avg = np.mean(v_matrices, axis=0)
    l_w = max(10, v_avg.shape[0] // 2)

    sub_matrices = [v_avg[i:i+l_w, :, 0].flatten() for i in range(v_avg.shape[0] - l_w + 1)]
    n_snapshots = len(sub_matrices)
    R_smooth = np.dot(np.array(sub_matrices).T.conj(), np.array(sub_matrices)) / n_snapshots

    vals, vecs = linalg.eigh(R_smooth)
    noise_sub = vecs[:, :-1]
    confidence = _signal_subspace_confidence(vals, n_snapshots)

    grid = np.linspace(-np.pi/2, np.pi/2, 181)
    spectrum = [1.0 / np.real(np.dot(np.dot(noise_sub.T.conj(), np.exp(-1j * np.pi * np.arange(R_smooth.shape[0]) * np.sin(th))).T.conj(), np.dot(noise_sub.T.conj(), np.exp(-1j * np.pi * np.arange(R_smooth.shape[0]) * np.sin(th))))) for th in grid]

    return float(np.degrees(grid[np.argmax(spectrum)])), confidence

def algo_iaa_apes(v_matrices, nt):
    """ IAA-APES Iterative solver. Single-snapshot execution for mobile hotspots. """
    v_snap = np.mean(v_matrices[-1], axis=0)[:, 0]
    m = len(v_snap)
    grid = np.linspace(-np.pi/2, np.pi/2, 91)
    p_spec = np.ones(len(grid))

    for _ in range(3):
        R = sum([p_spec[i] * np.dot(np.exp(-1j * np.pi * np.arange(m) * np.sin(th)).reshape(-1, 1), np.exp(-1j * np.pi * np.arange(m) * np.sin(th)).reshape(-1, 1).conj().T) for i, th in enumerate(grid)])
        R_inv = linalg.pinv(R + 1e-3 * np.eye(m))
        for i, th in enumerate(grid):
            a = np.exp(-1j * np.pi * np.arange(m) * np.sin(th)).reshape(-1, 1)
            denom = float(np.real(np.dot(np.dot(a.conj().T, R_inv), a)[0, 0]))
            if denom > 0: p_spec[i] = np.abs(np.dot(np.dot(a.conj().T, R_inv), v_snap.reshape(-1, 1))[0, 0] / denom)**2

    # Single-snapshot method: there is no eigenvalue spectrum to gate on here,
    # and it is only ever selected for the already-sparse-data branch (see the
    # dispatcher's STARVED_LIMIT routing), so its own reliability is already
    # signaled by why it was chosen rather than by a distinct data-driven check.
    return float(np.degrees(grid[np.argmax(p_spec)])), "MEDIUM"

def algo_res_2d_music(v_matrices, nt):
    """ Residual 2D MUSIC for clean arrays >= 3x3 """
    v_avg = np.mean(v_matrices, axis=0)
    z = v_avg[:, :, 0].T
    n_snapshots = v_matrices.shape[1]
    R = np.dot(z, np.conj(z).T) / n_snapshots

    vals, vecs = linalg.eigh(R)
    noise_sub = vecs[:, :-1]
    confidence = _signal_subspace_confidence(vals, n_snapshots)

    grid = np.linspace(-np.pi/2, np.pi/2, 181)
    spectrum = [1.0 / np.real(np.dot(np.dot(noise_sub.T.conj(), np.exp(-1j * np.pi * np.arange(nt) * np.sin(th))).T.conj(), np.dot(noise_sub.T.conj(), np.exp(-1j * np.pi * np.arange(nt) * np.sin(th))))) for th in grid]

    return float(np.degrees(grid[np.argmax(spectrum)])), confidence

SSE_REGISTRY = {
    "CA_ESPRIT": algo_ca_esprit,
    "SPOTFI": algo_spotfi,
    "IAA_APES": algo_iaa_apes,
    "RES_2D_MUSIC": algo_res_2d_music
}
