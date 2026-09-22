#!/usr/bin/env python3
"""
Module: aoa.py
Angle-of-departure estimation from beamforming feedback.

Estimators consume a covariance matrix and are parameterised by array geometry
as element positions in metres. A method requiring a uniform linear array
declares it rather than assuming it.

Covariance (Itahara et al., IEEE Access 2022, Prop. 1):

    C = (1/K) sum_k W_k V_k diag(L) V_k^H W_k^H

L is the per-stream gain, W_k an optional per-antenna phase calibration. Its
signal subspace is spanned by the AoD steering vectors, so MUSIC applies.

Two properties follow. It uses the whole V_k weighted by stream gain, not one
column. And it is invariant to the standard's phase gauge: the report fixes V's
last row real non-negative, a choice of D in V -> V D for diagonal unitary D,
and D diag(L) D^H = diag(L) identically. Estimators therefore take a
covariance, never V columns as snapshots.

Sign convention follows the same model: a(x) = exp(j 2 pi d sin x / lambda),
H = A(theta) R A(phi)^H, so V spans the column space of A(phi) and the steering
vector is unconjugated.

Absolute time of flight is absent: a delay multiplies H(f) by one scalar per
subcarrier, which cannot change right singular vectors. Relative delay between
paths survives and is what joint_aod_delay estimates.
"""
import numpy as np

_C_LIGHT = 299792458.0


# ---------------------------------------------------------------- geometry

def uniform_linear(n_elements, spacing_m):
    """Element positions for a uniform linear array along x, as (n, 2) metres."""
    return np.stack([np.arange(n_elements) * float(spacing_m),
                     np.zeros(n_elements)], axis=1)


def is_uniform_linear(positions_m, tolerance=1e-9):
    """
    True when positions are collinear and equally spaced.

    Required by ESPRIT's shift invariance and by spatial smoothing. Checked so
    those methods can refuse geometry they cannot handle.
    """
    p = np.asarray(positions_m, dtype=float)
    if len(p) < 2:
        return False
    delta = np.diff(p, axis=0)
    if not np.allclose(delta, delta[0], atol=tolerance):
        return False
    return np.linalg.norm(delta[0]) > tolerance


def steering_matrix(positions_m, wavelength_m, angles_rad):
    """
    Steering vectors for the given bearings, one row per bearing.

        a_m(phi) = exp(+j 2 pi (p_m . u(phi)) / lambda),  u = (sin phi, cos phi)

    Bearings run from +y toward +x, so a ULA along x reduces to
    exp(j 2 pi m d sin phi / lambda).
    """
    p = np.asarray(positions_m, dtype=float)
    angles = np.atleast_1d(np.asarray(angles_rad, dtype=float))
    direction = np.stack([np.sin(angles), np.cos(angles)], axis=1)   # (G, 2)
    projection = direction @ p.T                                     # (G, n)
    return np.exp(2j * np.pi * projection / float(wavelength_m))


def ambiguity(positions_m, wavelength_m, grid_rad=None):
    """
    Largest normalised sidelobe outside the main lobe, in [0, 1].

    Near 1 means distinct bearings produce near-identical steering vectors and
    no estimator can separate them. Depends on geometry and frequency only.
    """
    if grid_rad is None:
        grid_rad = np.linspace(-np.pi / 2, np.pi / 2, 721)
    a = steering_matrix(positions_m, wavelength_m, grid_rad)
    reference = steering_matrix(positions_m, wavelength_m, [0.0])[0]
    response = np.abs(a @ reference.conj()) / len(reference)

    # Main lobe runs to the first null either side; its width depends on
    # aperture in wavelengths, so it is found rather than assumed.
    centre = int(np.argmin(np.abs(grid_rad)))
    left = centre
    while left > 0 and response[left - 1] < response[left]:
        left -= 1
    right = centre
    while right < len(response) - 1 and response[right + 1] < response[right]:
        right += 1
    outside = np.concatenate([response[:left], response[right + 1:]])
    return float(outside.max()) if outside.size else 0.0


def sensitivity(positions_m, wavelength_m, angle_rad=0.0, delta_rad=1e-4):
    """
    Rate of change of the steering vector with bearing. Small means nearby
    bearings are hard to separate. Geometry and frequency only.
    """
    a0 = steering_matrix(positions_m, wavelength_m, [angle_rad])[0]
    a1 = steering_matrix(positions_m, wavelength_m, [angle_rad + delta_rad])[0]
    return float(np.linalg.norm(a1 - a0) / delta_rad)


# ------------------------------------------------------------- covariance

def bff_covariance(v_stack, stream_gain_db, calibration=None):
    """
    Covariance from one or more reports (Itahara et al. eq. 14-15).

    v_stack          (P, K, N, Nc) complex, P reports of K subcarriers
    stream_gain_db   (P, Nc) Average SNR field, dB; the subcarrier-averaged
                     stream gain the derivation requires
    calibration      optional (N,) or (K, N) per-antenna phase correction

    Returns (N, N), gauge-invariant. Gains convert to linear because the
    derivation treats them as powers.
    """
    v = np.asarray(v_stack)
    if v.ndim == 3:
        v = v[None, ...]
    gains_db = np.atleast_2d(np.asarray(stream_gain_db, dtype=float))
    power = 10.0 ** (gains_db / 10.0)

    n_reports, n_sub, n_ant, n_streams = v.shape
    if power.shape != (n_reports, n_streams):
        raise ValueError(f"stream gains {power.shape} do not match reports "
                         f"{(n_reports, n_streams)}")

    if calibration is not None:
        w = np.asarray(calibration, dtype=complex)
        if w.ndim == 1:
            w = np.broadcast_to(w, (n_sub, n_ant))
        v = v * w[None, :, :, None]

    total = np.zeros((n_ant, n_ant), dtype=complex)
    for i in range(n_reports):
        weighted = v[i] * power[i][None, None, :]           # (K, N, Nc)
        total += np.einsum('kij,klj->il', weighted, v[i].conj()) / n_sub
    covariance = total / n_reports
    # Hermitian by construction; symmetrise against accumulated rounding so
    # eigh does not see a matrix it must silently repair.
    return (covariance + covariance.conj().T) / 2.0


def spatial_smooth(covariance, sub_size):
    """
    Forward-backward spatial smoothing over overlapping sub-arrays.

    Decorrelates coherent paths, which reflections are; without it two
    phase-synchronised arrivals read as one source at a false bearing.

    Requires a uniform linear array and costs aperture: sub-array size M leaves
    at most M-1 resolvable sources. The caller checks geometry.
    """
    c = np.asarray(covariance)
    n = c.shape[0]
    if not 1 < sub_size <= n:
        raise ValueError(f"sub-array size {sub_size} outside 2..{n}")
    forward = np.zeros((sub_size, sub_size), dtype=complex)
    for j in range(n - sub_size + 1):
        forward += c[j:j + sub_size, j:j + sub_size]
    forward /= (n - sub_size + 1)
    # Backward average: the reversed, conjugated covariance of a ULA carries the
    # same angle information, so including it doubles the effective sub-arrays
    # without further shrinking the aperture.
    exchange = np.eye(sub_size)[::-1]
    backward = exchange @ forward.conj() @ exchange
    return (forward + backward) / 2.0


def mdl_source_count(eigenvalues_desc, n_snapshots, max_sources=None):
    """
    Source count by minimum description length.
    Wax & Kailath, IEEE Trans. ASSP 33(2):387-392, 1985.

    Scores the flatness of the eigenvalue tail under each k-source hypothesis
    against a penalty growing with k. No free parameter. The count is returned
    alongside its eigenvalues so a caller can override it.
    """
    values = np.sort(np.asarray(eigenvalues_desc, dtype=float))[::-1]
    m = len(values)
    limit = m - 1 if max_sources is None else min(max_sources, m - 1)
    best_k, best_score = 0, np.inf
    for k in range(0, limit + 1):
        tail = np.clip(values[k:], 1e-300, None)
        if len(tail) < 2:
            break
        geometric = np.exp(np.mean(np.log(tail)))
        arithmetic = np.mean(tail)
        likelihood = (m - k) * n_snapshots * np.log(arithmetic / geometric)
        penalty = 0.5 * k * (2 * m - k) * np.log(n_snapshots)
        score = likelihood + penalty
        if score < best_score:
            best_k, best_score = k, score
    return best_k


# -------------------------------------------------------------- candidates

def _local_maxima(spectrum, grid_rad):
    """Interior local maxima, strongest first."""
    s = np.asarray(spectrum, dtype=float)
    interior = np.where((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]))[0] + 1
    order = interior[np.argsort(s[interior])[::-1]]
    return [(float(grid_rad[i]), float(s[i])) for i in order]


def _result(name, grid_rad, spectrum, eigenvalues, n_sources, extra=None):
    """
    Common estimator return shape: full spectrum and ranked candidate set,
    never a single angle. Pruning is a later decision.
    """
    out = {
        "estimator": name,
        "grid_rad": np.asarray(grid_rad),
        "spectrum": np.asarray(spectrum, dtype=float),
        "candidates_rad": _local_maxima(spectrum, grid_rad),
        "eigenvalues": (None if eigenvalues is None
                        else np.sort(np.asarray(eigenvalues, dtype=float))[::-1]),
        "n_sources": n_sources,
    }
    if extra:
        out.update(extra)
    return out


# -------------------------------------------------------------- estimators

def music(covariance, positions_m, wavelength_m, n_sources=None,
          grid_rad=None, n_snapshots=1, smoothing=None):
    """
    Spectral MUSIC on the BFI covariance, arbitrary geometry.

    Spectral rather than root form: rooting in z = exp(-j pi sin theta) needs
    element k to contribute z^k, which fixes uniform spacing and accepts no
    element positions. The grid search matches it on a ULA and generalises.
    """
    if grid_rad is None:
        grid_rad = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    c = np.asarray(covariance)
    positions = np.asarray(positions_m, dtype=float)

    if smoothing:
        if not is_uniform_linear(positions):
            raise ValueError("spatial smoothing requires a uniform linear array")
        c = spatial_smooth(c, smoothing)
        positions = positions[:smoothing]

    values, vectors = np.linalg.eigh(c)                 # ascending
    descending = values[::-1]
    if n_sources is None:
        n_sources = max(1, mdl_source_count(descending, n_snapshots,
                                            max_sources=len(values) - 1))
    n_sources = int(np.clip(n_sources, 1, len(values) - 1))

    noise = vectors[:, :len(values) - n_sources]        # smallest eigenvalues
    a = steering_matrix(positions, wavelength_m, grid_rad)
    projected = a.conj() @ noise
    denominator = np.einsum('gi,gi->g', projected, projected.conj()).real
    spectrum = 1.0 / np.maximum(denominator, 1e-300)
    return _result("music", grid_rad, spectrum, descending, n_sources)


def esprit(covariance, positions_m, wavelength_m, n_sources=None,
           n_snapshots=1, grid_rad=None):
    """
    ESPRIT on the BFI covariance. Closed form, no grid.

    Requires a uniform linear array: it rests on one sub-array being a rigid
    translation of another. Refused otherwise rather than approximated.

    The returned spectrum is spikes on the shared grid, for a common result
    shape; the angles themselves are exact.
    """
    positions = np.asarray(positions_m, dtype=float)
    if not is_uniform_linear(positions):
        raise ValueError("esprit requires a uniform linear array")
    if grid_rad is None:
        grid_rad = np.linspace(-np.pi / 2, np.pi / 2, 1801)

    spacing = float(np.linalg.norm(positions[1] - positions[0]))
    c = np.asarray(covariance)
    values, vectors = np.linalg.eigh(c)
    descending = values[::-1]
    if n_sources is None:
        n_sources = max(1, mdl_source_count(descending, n_snapshots,
                                            max_sources=len(values) - 1))
    n_sources = int(np.clip(n_sources, 1, len(values) - 1))

    signal = vectors[:, -n_sources:]
    upper, lower = signal[:-1, :], signal[1:, :]
    psi = np.linalg.pinv(upper) @ lower
    # a_{m+1}/a_m = exp(+j 2 pi d sin(phi) / lambda) under this module's sign
    # convention, so the rotation's argument carries sin(phi) directly.
    phases = np.angle(np.linalg.eigvals(psi))
    sines = np.clip(phases * wavelength_m / (2 * np.pi * spacing), -1.0, 1.0)
    angles = np.arcsin(sines)

    spectrum = np.zeros_like(grid_rad)
    for angle in angles:
        spectrum[np.argmin(np.abs(grid_rad - angle))] += 1.0
    return _result("esprit", grid_rad, spectrum, descending, n_sources,
                   {"candidates_rad": [(float(a), 1.0) for a in np.sort(angles)]})


def spice(covariance, positions_m, wavelength_m, grid_rad=None,
          iterations=15, n_sources=None, n_snapshots=1):
    """
    SPICE, sparse iterative covariance-based estimation.
    Stoica, Babu & Li, IEEE Trans. Signal Processing 59(2):629-638, 2011.

    Fits sum_g p_g a_g a_g^H + noise by a weighted covariance-matching
    criterion with a closed-form multiplicative update: no step size,
    regularisation weight or stopping threshold.

    Used in place of IAA-APES because it fits a covariance. IAA's amplitude
    step reads data snapshots, whose phase the standard's gauge sets rather
    than the channel.
    """
    if grid_rad is None:
        grid_rad = np.linspace(-np.pi / 2, np.pi / 2, 721)
    c = np.asarray(covariance)
    n = c.shape[0]
    # steering_matrix returns (G, n); the plain transpose gives the (n, G)
    # dictionary. Conjugating as well negates the phase and mirrors every
    # estimate.
    a = steering_matrix(positions_m, wavelength_m, grid_rad).T         # (n, G)
    a /= np.linalg.norm(a, axis=0, keepdims=True)
    n_grid = a.shape[1]

    power = np.abs(np.einsum('ig,ij,jg->g', a.conj(), c, a)).real / n
    power = np.maximum(power, 1e-12)
    noise = max(float(np.real(np.trace(c)) / n) * 1e-3, 1e-12)

    weights = np.concatenate([np.ones(n_grid), np.full(n, 1.0)])
    dictionary = np.concatenate([a, np.eye(n)], axis=1)                # (n, G+n)
    gamma = np.concatenate([power, np.full(n, noise)])

    for _ in range(iterations):
        r = (dictionary * gamma) @ dictionary.conj().T
        r_inv = np.linalg.pinv(r + 1e-12 * np.eye(n))
        # Closed-form update: each component scales by the ratio of the data
        # it explains to the model's prediction for it.
        numerator = np.einsum('ig,ij,jk,kl,lg->g', dictionary.conj(), r_inv,
                              c, r_inv, dictionary).real
        denominator = np.einsum('ig,ij,jg->g', dictionary.conj(), r_inv,
                                dictionary).real
        numerator = np.maximum(numerator, 0.0)
        denominator = np.maximum(denominator, 1e-300)
        scale = np.sqrt(numerator / denominator) / np.sqrt(weights)
        gamma = gamma * scale
        gamma = np.maximum(gamma, 1e-300)

    values = np.linalg.eigvalsh(c)[::-1]
    if n_sources is None:
        n_sources = max(1, mdl_source_count(values, n_snapshots,
                                            max_sources=n - 1))
    return _result("spice", grid_rad, gamma[:n_grid], values, int(n_sources),
                   {"noise_estimate": gamma[n_grid:].copy()})


def joint_aod_delay(v_stack, stream_gain_db, positions_m, frequencies_hz,
                    delay_grid_s=None, grid_rad=None, n_sources=None,
                    calibration=None):
    """
    Joint bearing and relative delay, by MUSIC over a 2-D manifold.

    Absolute delay is not estimable and is not attempted. What survives is one
    path's delay relative to another, which changes how paths mix across
    frequency.

    The covariance is formed over a stacked antenna-frequency manifold so each
    subcarrier block carries its own phase reference. Delays are relative to
    the strongest path, so any delay common to every path cancels.

    Delay resolution is set by bandwidth, roughly 1/B: about 25 ns at 40 MHz.
    """
    if grid_rad is None:
        grid_rad = np.linspace(-np.pi / 2, np.pi / 2, 181)
    if delay_grid_s is None:
        delay_grid_s = np.linspace(-50e-9, 50e-9, 101)

    v = np.asarray(v_stack)
    if v.ndim == 3:
        v = v[None, ...]
    frequencies = np.asarray(frequencies_hz, dtype=float)
    positions = np.asarray(positions_m, dtype=float)
    n_reports, n_sub, n_ant, n_streams = v.shape
    gains = 10.0 ** (np.atleast_2d(np.asarray(stream_gain_db, float)) / 10.0)

    # Per-subcarrier covariances, each gauge-invariant, stacked so frequency
    # is a modelled dimension.
    blocks = np.zeros((n_sub, n_ant, n_ant), dtype=complex)
    for i in range(n_reports):
        w = v[i]
        if calibration is not None:
            cal = np.asarray(calibration, dtype=complex)
            if cal.ndim == 1:
                cal = np.broadcast_to(cal, (n_sub, n_ant))
            w = w * cal[:, :, None]
        blocks += np.einsum('kij,j,klj->kil', w, gains[i], w.conj())
    blocks /= n_reports

    # Dominant per-subcarrier direction taken from the covariance, not V, so
    # it is gauge-free; its residual global phase is fixed by a real reference.
    signature = np.zeros((n_sub, n_ant), dtype=complex)
    for k in range(n_sub):
        values, vectors = np.linalg.eigh(blocks[k])
        principal = vectors[:, -1] * np.sqrt(max(values[-1], 0.0))
        reference = principal[np.argmax(np.abs(principal))]
        signature[k] = principal * (np.abs(reference) / reference)

    stacked = signature.reshape(-1)                                  # (K*N,)
    covariance = np.outer(stacked, stacked.conj())
    values, vectors = np.linalg.eigh(covariance)
    n_sources = 1 if n_sources is None else int(n_sources)
    noise = vectors[:, :max(1, covariance.shape[0] - n_sources)]

    centre = frequencies.mean()
    spectrum = np.zeros((len(grid_rad), len(delay_grid_s)))
    for gi, angle in enumerate(grid_rad):
        per_sub = np.stack([
            steering_matrix(positions, _C_LIGHT / f, [angle])[0]
            for f in frequencies])                                   # (K, N)
        for di, delay in enumerate(delay_grid_s):
            ramp = np.exp(-2j * np.pi * (frequencies - centre) * delay)
            manifold = (per_sub * ramp[:, None]).reshape(-1)
            manifold /= np.linalg.norm(manifold)
            projection = manifold.conj() @ noise
            spectrum[gi, di] = 1.0 / max(float(np.vdot(projection, projection).real),
                                         1e-300)

    marginal = spectrum.max(axis=1)
    best_delay = delay_grid_s[np.argmax(spectrum, axis=1)]
    return _result("joint_aod_delay", grid_rad, marginal, values[::-1],
                   n_sources,
                   {"spectrum_2d": spectrum,
                    "delay_grid_s": np.asarray(delay_grid_s),
                    "delay_at_peak_s": best_delay})


# ------------------------------------------------------------ capabilities

# min_reports is the count an estimator structurally needs. A single report is
# not a single snapshot: it carries V on every subcarrier and bff_covariance
# averages over them, so the covariance is full rank from one packet and every
# entry declares 1. Nothing here is gated on rank.
#
# The count a deployment gates on for precision is MIN_REPORTS_<NAME> in
# config.env, which states its basis; dispatch.report_gates() reads it and
# refuses a value below the one declared here.
#
# joint_aod_delay has no precision curve; at ~826 ms per solve it dominates
# bench_precision.py.

ESTIMATORS = {
    "music": {
        "function": music,
        "input": "covariance",
        "needs_uniform_linear": False,
        "needs_geometry": True,
        "min_reports": 1,
        "handles_coherent": False,          # only with smoothing, which needs a ULA
        "uses_frequency_dimension": False,
        "reference": "Schmidt 1986; Itahara et al., IEEE Access 2022 (BFI form)",
    },
    "esprit": {
        "function": esprit,
        "input": "covariance",
        "needs_uniform_linear": True,
        "needs_geometry": True,
        "min_reports": 1,
        "handles_coherent": False,
        "uses_frequency_dimension": False,
        "reference": "Roy & Kailath, IEEE Trans. ASSP 37(7):984-995, 1989",
    },
    "spice": {
        "function": spice,
        "input": "covariance",
        "needs_uniform_linear": False,
        "needs_geometry": True,
        "min_reports": 1,
        "handles_coherent": True,
        "uses_frequency_dimension": False,
        "reference": "Stoica, Babu & Li, IEEE TSP 59(2):629-638, 2011",
    },
    "joint_aod_delay": {
        "function": joint_aod_delay,
        "input": "reports",
        "needs_uniform_linear": False,
        "needs_geometry": True,
        "min_reports": 1,
        "handles_coherent": True,
        "uses_frequency_dimension": True,
        "reference": "Kotaru et al., SIGCOMM 2015 (2-D AoA/ToF), adapted to "
                     "relative delay",
    },
}
