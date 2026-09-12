#!/usr/bin/env python3
"""
Module: 3_1_Spatial_Algorithms.py
Spatial Subspace Extractor (SSE) angle-of-departure estimators.

Every estimator returns (ap_aod_degrees, confidence). The eigen-decomposition
methods split the covariance matrix at rank 1, which only holds when one
propagation path dominates; indoor/through-wall multipath often breaks that,
and a rank-1 split on a two-path covariance returns whichever path is
momentarily strongest. The confidence flag reports whether the split was
justified for this chunk, and Stage 4 withholds the position update when it
is not.
"""
import os

import numpy as np
from scipy import linalg

# MDL needs two leftover noise eigenvalues to compare, so it is degenerate
# below 3 elements; the ratio check is used alone there.
_MDL_MIN_ELEMENTS = 3

# Both gates are env-overridable and NEITHER is calibrated against labelled
# captures. They disagree on real data: the bundled 11ac 3x1 trace gives
# eigenvalue ratios of 1.6-1.8 with MDL reporting one source, the 11ax 4x2
# trace gives 5.9-15.4 with MDL reporting two. Requiring both to pass rejects
# every chunk of both. Defaults below are placeholders pending calibration.
_EIGENVALUE_DOMINANCE_RATIO = float(os.getenv("SSE_EIGENVALUE_DOMINANCE_RATIO", 3.0))  # ~7 dB power gap
_MDL_MAX_SOURCES = int(os.getenv("SSE_MDL_MAX_SOURCES", 1))  # 2 accepts a weak second path


def _mdl_source_count(eigenvalues_desc, n_snapshots):
    """
    Estimate the number of sources by minimum description length.
    Wax & Kailath, IEEE Trans. ASSP 33(2):387-392, 1985.

    Under a k-source hypothesis the M-k smallest eigenvalues should all sit at
    the noise floor; MDL(k) scores the flatness of that tail against a penalty
    growing with k. k is searched only while M-k >= 2: with one eigenvalue left
    its geometric and arithmetic means are trivially equal and the hypothesis
    always scores a perfect, meaningless fit.
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
    Return "HIGH" if a rank-1 split is justified for this covariance matrix,
    "LOW" if a second comparably-strong path is present.

    On a 2-element array MDL has only one testable hypothesis and cannot
    separate one path from several. That is an aperture limit, not an
    implementation gap; the ratio check runs at every array size.
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

    # Single snapshot, so no eigenvalue spectrum to gate on. Fixed MEDIUM: the
    # dispatcher only selects this on the starved-packet branch, so the reason
    # to distrust it is already known from the routing.
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
