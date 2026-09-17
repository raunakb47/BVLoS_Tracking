#!/usr/bin/env python3
"""
Module: dispatch.py
Stage 3. Select the estimators a bucket's data supports, run them, and append
the results to the log.

Selection is a capability match, not a quality judgement. Each entry in
aoa.ESTIMATORS declares what it requires -- uniform linear geometry, a minimum
report count, tolerance of coherent sources -- and those are compared against
facts decoded in Stage 2. An estimator is either applicable or not, and both
outcomes are recorded with the reason, so adding one is a single entry in
aoa.ESTIMATORS with no change here.

Geometry comes from site.json when present. Element positions are a property of
the hardware and no signal processing recovers them, so absent site.json a
nominal half-wavelength uniform linear array is used and every result names it
in geometry_source. A bearing against a nominal array carries an unknown
per-beamformer rotation; it is reproducible and stated, which is what makes a
later calibrated run able to test it.
"""
import json
import os
import sys

import numpy as np

import aoa
import observe

# Subcarrier spacing by standard, from the OFDM numerology: 802.11ac inherits
# 312.5 kHz, 802.11ax uses a four-times-denser 78.125 kHz grid. Needed to turn
# a subcarrier index into the frequency whose wavelength the steering uses.
SUBCARRIER_SPACING_HZ = {"AC": 312500.0, "AX": 78125.0}

_C_LIGHT = 299792458.0


def load_site(path):
    """
    Optional per-beamformer geometry:

        {"aa:bb:cc:dd:ee:ff": {"positions_m": [[0,0],[0.028,0],...],
                               "orientation_deg": 0.0}}

    Missing or unreadable, the caller falls back to a nominal array.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def geometry_for(beamformer, n_antennas, wavelength_m, site):
    """
    (positions, source, orientation_deg) for one beamformer.

    The nominal fallback is a half-wavelength uniform linear array, the spacing
    the standard's beamforming design assumes. A stated default, not a
    measurement; the source string records which was used.
    """
    entry = site.get(beamformer) or site.get(beamformer.lower())
    if entry and "positions_m" in entry:
        return (np.asarray(entry["positions_m"], dtype=float),
                "site.json",
                float(entry.get("orientation_deg", 0.0)))
    return (aoa.uniform_linear(n_antennas, wavelength_m / 2.0),
            "nominal_half_wavelength_ula",
            0.0)


def subcarrier_frequencies(standard, bandwidth_mhz, centre_mhz, n_expected):
    """
    Absolute frequency of each reported subcarrier, or None if the set cannot
    be reconstructed.

    Uses Wi-BFI's subcarrier index table so the two cannot disagree about which
    subcarriers a report covers; a count mismatch means they have.
    """
    if centre_mhz is None:
        return None
    sys.path.insert(0, os.environ.get("WIBFI_DIR", "../Wi-BFI"))
    try:
        from main import subcarrier_indices
    except ImportError:
        return None
    indices = subcarrier_indices(standard, int(bandwidth_mhz))
    if indices is None or len(indices) != n_expected:
        return None
    spacing = SUBCARRIER_SPACING_HZ.get(standard)
    if spacing is None:
        return None
    return float(centre_mhz) * 1e6 + np.asarray(indices, dtype=float) * spacing


def coherence(covariances):
    """
    Similarity of consecutive per-report covariances, two ways.

    A diagnostic, not a gate: averaging reports is sound only while they
    describe the same spatial state, and how long that lasts is a property of
    the deployment.

    "matrix" compares whole covariances and includes the diagonal, which is
    per-antenna power and drifts slowly whatever the bearing does. "principal"
    compares dominant eigenvectors, which is what a bearing estimate reads.
    """
    if len(covariances) < 2:
        return None

    def principal(matrix):
        _, vectors = np.linalg.eigh(matrix)
        return vectors[:, -1]

    matrix_scores, principal_scores = [], []
    for a, b in zip(covariances[:-1], covariances[1:]):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            matrix_scores.append(abs(np.vdot(a, b)) / (na * nb))
            principal_scores.append(abs(np.vdot(principal(a), principal(b))))
    if not matrix_scores:
        return None
    return {"matrix": float(np.median(matrix_scores)),
            "principal": float(np.median(principal_scores))}


def applicable(name, spec, facts):
    """
    Whether an estimator can run on this bucket, and why not when it cannot.

    Every test compares a declared requirement against a decoded fact; none
    consults a measurement's value.
    """
    if spec["needs_uniform_linear"] and not facts["uniform_linear"]:
        return False, "geometry is not a uniform linear array"
    if facts["n_reports"] < spec["min_reports"]:
        return False, f"needs {spec['min_reports']} reports, bucket has {facts['n_reports']}"
    if spec["needs_geometry"] and facts["n_antennas"] < 2:
        return False, f"needs at least 2 elements, beamformer has {facts['n_antennas']}"
    if spec["uses_frequency_dimension"] and facts["frequencies_hz"] is None:
        return False, "subcarrier frequencies could not be reconstructed"
    # MUSIC and ESPRIT need at least one noise eigenvector.
    if name in ("music", "esprit") and facts["n_antennas"] < 2:
        return False, "no noise subspace on a single element"
    return True, None


def solve_bucket(records, log_prefix, site, max_reports=None):
    """
    Run every applicable estimator over one bucket, returning their results and
    the facts selection used.
    """
    records = sorted(records, key=lambda r: r["t"])
    if max_reports:
        records = records[-max_reports:]
    head = records[0]
    n_antennas = head["nr"]
    centre_mhz = head.get("freq_mhz")

    v_stack = np.stack([observe.read_v_matrix(r, log_prefix) for r in records])
    gains_db = np.array([r["stream_snr"] for r in records], dtype=float)

    wavelength = (_C_LIGHT / (centre_mhz * 1e6)) if centre_mhz else None
    positions, geometry_source, orientation = geometry_for(
        head["beamformer"], n_antennas,
        wavelength if wavelength else _C_LIGHT / 5.5e9, site)

    frequencies = subcarrier_frequencies(head["standard"], head["bw"],
                                         centre_mhz, head["nsubc"])

    per_report = [aoa.bff_covariance(v_stack[i:i + 1], gains_db[i:i + 1])
                  for i in range(len(records))]
    covariance = aoa.bff_covariance(v_stack, gains_db)

    facts = {
        "bucket": head["bucket"],
        "transmitter": head["transmitter"],
        "beamformer": head["beamformer"],
        "standard": head["standard"],
        "feedback": head["feedback"],
        "n_antennas": n_antennas,
        "n_streams": head["nc"],
        "n_subcarriers": head["nsubc"],
        "bandwidth_mhz": head["bw"],
        "centre_freq_mhz": centre_mhz,
        "grouping_ng": head["ng"],
        "codebook": head["codebook"],
        "n_reports": len(records),
        "time_span_s": records[-1]["t"] - records[0]["t"],
        "coherence": coherence(per_report),
        "uniform_linear": bool(aoa.is_uniform_linear(positions)),
        "geometry_source": geometry_source,
        "orientation_deg": orientation,
        "frequencies_hz": frequencies,
        "ambiguity": (aoa.ambiguity(positions, wavelength) if wavelength else None),
        "sensitivity": (aoa.sensitivity(positions, wavelength) if wavelength else None),
    }

    results = []
    for name, spec in aoa.ESTIMATORS.items():
        ok, reason = applicable(name, spec, facts)
        if not ok:
            results.append({"estimator": name, "ran": False, "reason": reason})
            continue
        try:
            if spec["input"] == "covariance":
                out = spec["function"](covariance, positions, wavelength,
                                       n_snapshots=head["nsubc"])
            else:
                out = spec["function"](v_stack, gains_db, positions, frequencies)
        except Exception as exc:                      # noqa: BLE001
            results.append({"estimator": name, "ran": False,
                            "reason": f"{type(exc).__name__}: {exc}"})
            continue
        candidates = [{"bearing_deg": float(np.degrees(a)), "strength": float(s)}
                      for a, s in out["candidates_rad"][:8]]
        entry = {
            "estimator": name,
            "ran": True,
            "n_sources": out["n_sources"],
            "candidates": candidates,
            "eigenvalues": (None if out["eigenvalues"] is None
                            else [float(v) for v in out["eigenvalues"]]),
        }
        if "delay_at_peak_s" in out and candidates:
            peak = int(np.argmin(np.abs(out["grid_rad"] -
                                        np.radians(candidates[0]["bearing_deg"]))))
            entry["relative_delay_s"] = float(out["delay_at_peak_s"][peak])
        results.append(entry)

    facts.pop("frequencies_hz")
    return facts, results


def solve(log_prefix, site_path=None, max_reports=None, out_path=None):
    """Run Stage 3 over every bucket in the log and append the results."""
    site = load_site(site_path)
    buckets = {}
    for record in observe.read_log(log_prefix):
        if record.get("kind") == "bfi":
            buckets.setdefault(record["bucket"], []).append(record)

    out_path = out_path or (log_prefix + ".solve.jsonl")
    written = 0
    with open(out_path, "a") as handle:
        for key in sorted(buckets, key=lambda k: -len(buckets[k])):
            facts, results = solve_bucket(buckets[key], log_prefix, site,
                                          max_reports)
            handle.write(json.dumps({"kind": "solve", "facts": facts,
                                     "results": results}) + "\n")
            written += 1
    return out_path, written


def report(solve_path, stream=sys.stdout):
    """
    Print the solve file as a table, readable without parsing JSON.

    Estimators are shown side by side rather than reconciled: agreement means
    the data supports a bearing, spread means it does not, and a single
    reconciled number would hide which.

    Bearings are relative to the assumed array. Against a nominal array they
    carry an unknown per-beamformer rotation, so what compares against a real
    room is the pattern across one beamformer's clients, not a single figure.
    """
    for line in open(solve_path):
        entry = json.loads(line)
        facts, results = entry["facts"], entry["results"]
        coh = facts.get("coherence") or {}
        print(f"\n{facts['transmitter']}  ->  {facts['beamformer']}", file=stream)
        print(f"   {facts['n_antennas']}x{facts['n_streams']} @ {facts['bandwidth_mhz']} MHz "
              f"{facts['standard']}/{facts['feedback']}   Ng={facts['grouping_ng']} "
              f"codebook={facts['codebook']}   {facts['centre_freq_mhz']} MHz",
              file=stream)
        span = facts["time_span_s"]
        print(f"   {facts['n_reports']} reports over {span:.0f} s"
              f"   coherence matrix={coh.get('matrix', float('nan')):.3f} "
              f"principal={coh.get('principal', float('nan')):.3f}", file=stream)
        print(f"   geometry {facts['geometry_source']}"
              f"   ambiguity {facts['ambiguity']:.2f}"
              f"   sensitivity {facts['sensitivity']:.1f}", file=stream)
        for result in results:
            if not result["ran"]:
                print(f"     {result['estimator']:17s} not run: {result['reason']}",
                      file=stream)
                continue
            bearings = "  ".join(f"{c['bearing_deg']:+7.1f}"
                                 for c in result["candidates"][:4])
            extra = ""
            if "relative_delay_s" in result:
                extra = f"   relative delay {result['relative_delay_s'] * 1e9:+.0f} ns"
            print(f"     {result['estimator']:17s} L={result['n_sources']}"
                  f"  {bearings}{extra}", file=stream)
            # A linear array cannot separate a bearing from its mirror about
            # the array axis, so each candidate names two directions.
            if result["candidates"]:
                lead = result["candidates"][0]["bearing_deg"]
                print(f"     {'':17s}    leading candidate is {lead:+.1f} deg "
                      f"OR its mirror {180.0 - lead:+.1f} deg", file=stream)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: dispatch.py <log_prefix> [site.json] [max_reports]",
              file=sys.stderr)
        raise SystemExit(2)
    site_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else None
    path, count = solve(sys.argv[1], site_arg, cap)
    print(f"[*] {count} buckets solved -> {path}")
    report(path)
