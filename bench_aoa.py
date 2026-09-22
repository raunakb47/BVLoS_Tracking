#!/usr/bin/env python3
"""
Module: bench_aoa.py
Synthetic ground-truth bench for aoa.py.

Builds channels with known angles, applies the standard's compression -- SVD,
Givens quantisation, phase gauge -- and checks what each estimator recovers.
Every claim aoa.py's docstrings make is a test here: the sign convention, gauge
invariance of the covariance, absence of absolute delay and presence of relative
delay, recovery of known angles, and refusal of unsupported geometry.

Run: python3 bench_aoa.py
"""
import sys

import numpy as np

import aoa

C_LIGHT = 299792458.0


def channel(angles_deg, gains, positions_m, frequencies_hz, n_rx=2,
            rx_angles_deg=None, delays_s=None, noise=0.0, seed=0):
    """
    Per-subcarrier channel H_k = sum_l g_l e^{-j2pi f tau_l} a_rx a_tx^H
    (Itahara et al. eq. 3).
    """
    rng = np.random.default_rng(seed)
    if rx_angles_deg is None:
        rx_angles_deg = list(np.linspace(-20, 20, len(angles_deg)))
    if delays_s is None:
        delays_s = [0.0] * len(angles_deg)
    rx_positions = aoa.uniform_linear(n_rx, 0.025)
    n_tx = len(positions_m)
    out = np.zeros((len(frequencies_hz), n_rx, n_tx), dtype=complex)
    for k, f in enumerate(frequencies_hz):
        lam = C_LIGHT / f
        h = np.zeros((n_rx, n_tx), dtype=complex)
        for angle, gain, rx_angle, delay in zip(angles_deg, gains,
                                                rx_angles_deg, delays_s):
            a_tx = aoa.steering_matrix(positions_m, lam, [np.radians(angle)])[0]
            a_rx = aoa.steering_matrix(rx_positions, lam, [np.radians(rx_angle)])[0]
            h += gain * np.exp(-2j * np.pi * f * delay) * np.outer(a_rx, a_tx.conj())
        if noise:
            h += noise * (rng.standard_normal(h.shape) +
                          1j * rng.standard_normal(h.shape)) / np.sqrt(2)
        out[k] = h
    return out


def compress(h_stack, psi_bits=4, phi_bits=6, gauge=True):
    """
    The standard's compression: SVD, keep the leading Nc right singular
    vectors, quantise, fix the last row real non-negative.

    Quantisation is applied through the angles, where the standard applies it.
    """
    n_sub, n_rx, n_tx = h_stack.shape
    n_streams = min(n_rx, n_tx)
    v_out = np.zeros((n_sub, n_tx, n_streams), dtype=complex)
    gains = np.zeros((n_sub, n_streams))
    for k in range(n_sub):
        u, s, vh = np.linalg.svd(h_stack[k])
        v = vh.conj().T[:, :n_streams]
        if gauge:
            last = v[-1, :]
            v = v * (np.abs(last) / np.where(last == 0, 1.0, last))
        if psi_bits:
            # Stand-in for the Givens round trip: quantise phase at the
            # codebook's angle resolution.
            step = np.pi / (2 ** phi_bits)
            v = np.abs(v) * np.exp(1j * np.round(np.angle(v) / step) * step)
            if gauge:
                last = v[-1, :]
                v = v * (np.abs(last) / np.where(last == 0, 1.0, last))
            v, _ = np.linalg.qr(v)
            if gauge:
                last = v[-1, :]
                v = v * (np.abs(last) / np.where(last == 0, 1.0, last))
        v_out[k] = v
        gains[k] = s[:n_streams] ** 2
    return v_out, 10.0 * np.log10(np.maximum(gains.mean(axis=0), 1e-12))


def _peaks_deg(result, n):
    return sorted(np.degrees(a) for a, _ in result["candidates_rad"][:n])


def _report(name, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}: {detail}")
    return ok


def main():
    freqs = 5.18e9 + np.linspace(-10e6, 10e6, 52)
    lam0 = C_LIGHT / freqs.mean()
    ula4 = aoa.uniform_linear(4, 0.025)
    passed = []

    print("sign convention -- the model determines it, the bench confirms it")
    for truth in ([-40, 15], [25, -55], [10, 60]):
        h = channel(truth, [1.0, 0.5], ula4, freqs)
        v, g = compress(h)
        c = aoa.bff_covariance(v, g)
        got = _peaks_deg(aoa.music(c, ula4, lam0, n_sources=2), 2)
        err = max(abs(a - b) for a, b in zip(sorted(truth), got))
        passed.append(_report(f"music recovers {sorted(truth)}", err < 4.0,
                              f"got {[round(x, 1) for x in got]}, max error {err:.2f} deg"))

    print("\ngauge invariance -- the covariance must not depend on the convention")
    rng = np.random.default_rng(1)
    h = channel([20, -35], [1.0, 0.6], ula4, freqs)
    v, g = compress(h)
    d = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(v.shape[0], v.shape[2])))
    c_a = aoa.bff_covariance(v, g)
    c_b = aoa.bff_covariance(v * d[:, None, :], g)
    rel = np.abs(c_a - c_b).max() / np.abs(c_a).max()
    passed.append(_report("covariance invariant under re-gauge", rel < 1e-12,
                          f"relative difference {rel:.2e}"))

    print("\ndelay -- absolute must be absent, relative must be present")
    v0, g0 = compress(channel([25], [1.0], ula4, freqs))
    c0 = aoa.bff_covariance(v0, g0)
    worst = 0.0
    for tau in (1e-9, 100e-9, 1e-6):
        v1, g1 = compress(channel([25], [1.0], ula4, freqs, delays_s=[tau]))
        c1 = aoa.bff_covariance(v1, g1)
        worst = max(worst, np.abs(c1 - c0).max() / np.abs(c0).max())
    passed.append(_report("absolute delay leaves the covariance unchanged",
                          worst < 1e-9, f"largest relative change {worst:.2e}"))

    base = compress(channel([25, -30], [1.0, 0.6], ula4, freqs,
                            delays_s=[0.0, 0.0]))
    c_base = aoa.bff_covariance(*base)
    v2, g2 = compress(channel([25, -30], [1.0, 0.6], ula4, freqs,
                              delays_s=[0.0, 20e-9]))
    moved = np.abs(aoa.bff_covariance(v2, g2) - c_base).max() / np.abs(c_base).max()
    passed.append(_report("relative delay does change the covariance",
                          moved > 1e-3, f"relative change {moved:.2e}"))

    print("\nestimators against known angles")
    truth = [30, -20]
    h = channel(truth, [1.0, 0.55], ula4, freqs)
    v, g = compress(h)
    c = aoa.bff_covariance(v, g)

    got = _peaks_deg(aoa.music(c, ula4, lam0, n_sources=2), 2)
    err = max(abs(a - b) for a, b in zip(sorted(truth), got))
    passed.append(_report("music", err < 4.0,
                          f"{[round(x, 1) for x in got]} vs {sorted(truth)}, {err:.2f} deg"))

    r = aoa.esprit(c, ula4, lam0, n_sources=2)
    got = sorted(np.degrees(a) for a, _ in r["candidates_rad"])
    err = max(abs(a - b) for a, b in zip(sorted(truth), got))
    passed.append(_report("esprit", err < 6.0,
                          f"{[round(x, 1) for x in got]} vs {sorted(truth)}, {err:.2f} deg"))

    r = aoa.spice(c, ula4, lam0)
    got = _peaks_deg(r, 2)
    err = min(abs(got[0] - min(truth)), abs(got[0] - max(truth))) if got else 99
    passed.append(_report("spice finds a true bearing", err < 6.0,
                          f"strongest {[round(x, 1) for x in got[:2]]} vs {sorted(truth)}"))

    r = aoa.joint_aod_delay(v, g, ula4, freqs, n_sources=2)
    got = _peaks_deg(r, 2)
    err = min(abs(got[0] - min(truth)), abs(got[0] - max(truth))) if got else 99
    passed.append(_report("joint_aod_delay finds a true bearing", err < 8.0,
                          f"strongest {[round(x, 1) for x in got[:2]]} vs {sorted(truth)}"))

    print("\ngeometry refusals -- a method must decline what it cannot do")
    sparse = np.stack([[0.0, 0.0], [0.031, 0.0], [0.077, 0.0], [0.11, 0.0]])
    try:
        aoa.esprit(c, sparse, lam0, n_sources=1)
        passed.append(_report("esprit refuses a non-uniform array", False, "it did not"))
    except ValueError as exc:
        passed.append(_report("esprit refuses a non-uniform array", True, str(exc)))
    ok = not aoa.is_uniform_linear(sparse) and aoa.is_uniform_linear(ula4)
    passed.append(_report("is_uniform_linear discriminates", ok,
                          "uniform accepted, irregular rejected"))

    print("\ngeometry diagnostics, from positions and frequency alone")
    print(f"     lambda/2 ULA : ambiguity {aoa.ambiguity(ula4, lam0):.3f}   "
          f"sensitivity {aoa.sensitivity(ula4, lam0):.1f}")
    wide = aoa.uniform_linear(4, 0.10)
    print(f"     100 mm ULA   : ambiguity {aoa.ambiguity(wide, lam0):.3f}   "
          f"sensitivity {aoa.sensitivity(wide, lam0):.1f}")
    print("     (ambiguity near 1 means distinct bearings look alike: an aperture "
          "limit, not an estimator one)")

    print(f"\n{sum(passed)}/{len(passed)} checks passed")
    return 0 if all(passed) else 1


if __name__ == "__main__":
    sys.exit(main())
