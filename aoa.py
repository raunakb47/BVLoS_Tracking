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


def _grid_dictionary(covariance, positions_m, wavelength_m, grid_rad):
    """Covariance, (n, G) steering dictionary and grid shared by the power estimators."""
    if grid_rad is None:
        grid_rad = np.linspace(-np.pi / 2, np.pi / 2, 721)
    # steering_matrix returns (G, n); the plain transpose gives the dictionary.
    # Conjugating as well negates the phase and mirrors every estimate.
    a = steering_matrix(positions_m, wavelength_m, grid_rad).T
    return np.asarray(covariance), a, grid_rad


def _source_count(c, n_sources, n_snapshots):
    """Eigenvalues and MDL count, reported alongside; the power spectra do not use them."""
    values = np.linalg.eigvalsh(c)[::-1]
    if n_sources is None:
        n_sources = max(1, mdl_source_count(values, n_snapshots,
                                            max_sources=len(values) - 1))
    return values, int(n_sources)


def spice(covariance, positions_m, wavelength_m, tolerance, max_iterations,
          grid_rad=None, n_sources=None, n_snapshots=1):
    """
    SPICE with a separate noise power per element.
    Stoica, Babu & Li, IEEE TSP 59(2):629-638, 2011, eqs. (11), (21), (33)-(35).

    Criterion (13): powers on the grid and on the n canonical vectors are
    updated multiplicatively with fixed weights w_k = a_k^H Rhat^-1 a_k / n.
    The problem is convex and limit points are global solutions, so iteration
    stops on convergence: relative power change sum|dp| / sum p below
    tolerance, or max_iterations. Needs Rhat invertible, which bff_covariance
    gives from one report.
    """
    c, a, grid_rad = _grid_dictionary(covariance, positions_m, wavelength_m, grid_rad)
    n, n_grid = a.shape
    b = np.concatenate([a, np.eye(n)], axis=1)                          # (11)
    norm2 = np.sum(np.abs(b) ** 2, axis=0)
    p = np.einsum('ig,ij,jg->g', b.conj(), c, b).real / norm2 ** 2      # (35)
    root_w = np.sqrt(np.einsum('ig,ij,jg->g', b.conj(), np.linalg.inv(c), b).real / n)   # (21)
    w_vals, w_vecs = np.linalg.eigh(c)
    half = (w_vecs * np.sqrt(np.maximum(w_vals, 0.0))) @ w_vecs.conj().T
    for _ in range(max_iterations):
        r = (b * p) @ b.conj().T
        t = np.linalg.norm(b.conj().T @ np.linalg.inv(r) @ half, axis=1)   # ||a_k^H R^-1 Rhat^1/2||
        updated = p * t / (root_w * np.sum(root_w * p * t))              # (33), (34)
        change = np.abs(updated - p).sum() / p.sum()
        p = updated
        if change < tolerance:
            break
    values, n_sources = _source_count(c, n_sources, n_snapshots)
    return _result("spice", grid_rad, p[:n_grid], values, n_sources,
                   {"noise_estimate": p[n_grid:].copy()})


def samv2(covariance, positions_m, wavelength_m, iterations,
          grid_rad=None, n_sources=None, n_snapshots=1):
    """
    SAMV-2. Abeida, Zhang & Li, IEEE TSP 61(4):933-944, 2013, Table 1, eqs.
    (8), (9), (16).

    R = A P A^H + sigma I; p_k <- p_k (a^H R^-1 Rhat R^-1 a) / (a^H R^-1 a),
    sigma <- tr(R^-2 Rhat) / tr(R^-2). A fixed point satisfies
    a^H R^-1 Rhat R^-1 a = a^H R^-1 a, the stochastic ML stationarity
    condition for p_k.
    """
    c, a, grid_rad = _grid_dictionary(covariance, positions_m, wavelength_m, grid_rad)
    n = a.shape[0]
    p = np.einsum('ig,ij,jg->g', a.conj(), c, a).real / np.sum(np.abs(a) ** 2, axis=0) ** 2   # (8)
    sigma = float(np.trace(c).real) / n                                  # (9)
    for _ in range(iterations):
        r_inv = np.linalg.inv((a * p) @ a.conj().T + sigma * np.eye(n))
        ria = r_inv @ a
        num = np.einsum('ig,ij,jg->g', ria.conj(), c, ria).real
        den = np.einsum('ig,ig->g', a.conj(), ria).real
        p = p * num / den                                                # (16)
        r_inv2 = r_inv @ r_inv
        sigma = float(np.trace(r_inv2 @ c).real / np.trace(r_inv2).real)   # (16)
    values, n_sources = _source_count(c, n_sources, n_snapshots)
    return _result("samv2", grid_rad, p, values, n_sources, {"noise_estimate": sigma})


def iaa(covariance, positions_m, wavelength_m, iterations,
        grid_rad=None, n_sources=None, n_snapshots=1):
    """
    IAA-APES power spectrum. Yardibi, Li, Stoica, Xue & Baggeroer, IEEE TAES
    46(1):425-443, 2010, Table II.

    R = A P A^H with no noise term; P_k = (1/N) sum_n |a^H R^-1 y(n)|^2 /
    (a^H R^-1 a)^2, written through Rhat = (1/N) sum_n y(n) y(n)^H so it takes
    the gauge-invariant covariance. The amplitude estimates are not formed.
    """
    c, a, grid_rad = _grid_dictionary(covariance, positions_m, wavelength_m, grid_rad)
    p = np.einsum('ig,ij,jg->g', a.conj(), c, a).real / np.sum(np.abs(a) ** 2, axis=0) ** 2
    for _ in range(iterations):
        # pinv: A P A^H is singular when few p_k are non-zero.
        ria = np.linalg.pinv((a * p) @ a.conj().T) @ a
        den = np.einsum('ig,ig->g', a.conj(), ria).real
        p = np.einsum('ig,ij,jg->g', ria.conj(), c, ria).real / den ** 2
    values, n_sources = _source_count(c, n_sources, n_snapshots)
    return _result("iaa", grid_rad, p, values, n_sources)


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
#
# params are passed to the function. Iteration settings for spice, samv2 and
# iaa, measured on the AWUS036AXM captures (9 buckets; full data and 1-, 5-
# and 20-report subsets; 225 covariances) against a 5000-iteration answer:
#
#   spice  stops below 3e-5 relative power change, at most 5000 iterations.
#          Leading bearing equal to the 5000-iteration answer in 100% of
#          cases, top three in 93%; median 719 iterations, ~0.1 ms each at
#          n=4. A fixed 15 settled none.
#   samv2  15 iterations. The SAMV paper states no count; 15 is the IAA
#          paper's (Table II). Not converged at 15 or at 5000 on these
#          captures: the leading bearing settles by iteration 150 (median)
#          and 1500 (90%).
#   iaa    15 iterations, IAA paper Table II. Settled by 5 in every case.
#
# The capture-derived values describe this hardware class; re-derive them on
# another.

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
    # Off the active set, kept for trial: ULA only, and shares eigh(C) with
    # music. Restore the entry to enable it.
    # "esprit": {
    #     "function": esprit,
    #     "input": "covariance",
    #     "needs_uniform_linear": True,
    #     "needs_geometry": True,
    #     "min_reports": 1,
    #     "handles_coherent": False,
    #     "uses_frequency_dimension": False,
    #     "reference": "Roy & Kailath, IEEE Trans. ASSP 37(7):984-995, 1989",
    # },
    "spice": {
        "function": spice,
        "input": "covariance",
        "params": {"tolerance": 3e-5, "max_iterations": 5000},
        "needs_uniform_linear": False,
        "needs_geometry": True,
        "min_reports": 1,
        "handles_coherent": True,
        "uses_frequency_dimension": False,
        "reference": "Stoica, Babu & Li, IEEE TSP 59(2):629-638, 2011, eq. (33)",
    },
    "samv2": {
        "function": samv2,
        "input": "covariance",
        "params": {"iterations": 15},
        "needs_uniform_linear": False,
        "needs_geometry": True,
        "min_reports": 1,
        "handles_coherent": True,
        "uses_frequency_dimension": False,
        "reference": "Abeida, Zhang & Li, IEEE TSP 61(4):933-944, 2013, eq. (16)",
    },
    "iaa": {
        "function": iaa,
        "input": "covariance",
        "params": {"iterations": 15},
        "needs_uniform_linear": False,
        "needs_geometry": True,
        "min_reports": 1,
        "handles_coherent": True,
        "uses_frequency_dimension": False,
        "reference": "Yardibi, Li, Stoica, Xue & Baggeroer, IEEE TAES 46(1):425-443, "
                     "2010, Table II",
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
