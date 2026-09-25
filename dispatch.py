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
aoa.ESTIMATORS with no change here. The report count gated on is set per
deployment in config.env; see report_gates().

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
import time

import numpy as np

import aoa
import observe

# Subcarrier spacing by standard, from the OFDM numerology: 802.11ac inherits
# 312.5 kHz, 802.11ax uses a four-times-denser 78.125 kHz grid. Needed to turn
# a subcarrier index into the frequency whose wavelength the steering uses.
SUBCARRIER_SPACING_HZ = {"AC": 312500.0, "AX": 78125.0}

_C_LIGHT = 299792458.0

_GATE_PREFIX = "MIN_REPORTS_"


def report_gates(environ=None):
    """
    Minimum report count per estimator, from MIN_REPORTS_<NAME> in config.env.

    How many reports a bearing needs before it stops moving is a property of
    the hardware and the room, so it is configured, not coded. Unset, the count
    the estimator declares in aoa.ESTIMATORS applies. A variable naming no
    estimator, not an integer, or below the declared count raises: each would
    otherwise leave a gate silently unapplied.
    """
    environ = os.environ if environ is None else environ
    gates = {name: spec["min_reports"] for name, spec in aoa.ESTIMATORS.items()}
    for key, value in environ.items():
        if not key.startswith(_GATE_PREFIX):
            continue
        name = key[len(_GATE_PREFIX):].lower()
        if name not in gates:
            raise ValueError(f"{key} names no estimator; expected one of "
                             + ", ".join(_GATE_PREFIX + n.upper() for n in gates))
        try:
            count = int(value)
        except ValueError:
            raise ValueError(f"{key}={value!r} is not an integer") from None
        declared = aoa.ESTIMATORS[name]["min_reports"]
        if count < declared:
            raise ValueError(f"{key}={count} is below the {declared} "
                             f"report(s) {name} requires")
        gates[name] = count
    return gates


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


def subcarrier_frequencies(standard, bandwidth_mhz, centre_mhz, n_expected, ng=None):
    """
    Absolute frequency of each reported subcarrier, or None if the set cannot
    be reconstructed.

    Uses Wi-BFI's subcarrier index table so the two cannot disagree about which
    subcarriers a report covers; a count mismatch means they have. Positions
    exist only at the native grouping, so any other ng returns None rather than
    a set the standard does not define.
    """
    if centre_mhz is None:
        return None
    sys.path.insert(0, os.environ.get("WIBFI_DIR", "../Wi-BFI"))
    try:
        from main import NATIVE_NG, subcarrier_indices
    except ImportError:
        return None
    if ng is not None and ng != NATIVE_NG.get(standard):
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


def _is_grouped(standard, ng):
    """
    Whether a report covers fewer subcarriers than the standard's full set.

    True when Ng is unknown: an undecoded grouping is not evidence of the
    native one, and the frequency-dependent estimators must not run on it.
    """
    sys.path.insert(0, os.environ.get("WIBFI_DIR", "../Wi-BFI"))
    try:
        from main import NATIVE_NG
    except ImportError:
        return True
    return ng != NATIVE_NG.get(standard)


def applicable(name, spec, facts, min_reports):
    """
    Whether an estimator can run on this bucket, and why not when it cannot.

    Every test compares a declared or configured requirement against a decoded
    fact; none consults a measurement's value.
    """
    if spec["needs_uniform_linear"] and not facts["uniform_linear"]:
        return False, "geometry is not a uniform linear array"
    if facts["n_reports"] < min_reports:
        return False, f"needs {min_reports} reports, bucket has {facts['n_reports']}"
    if spec["needs_geometry"] and facts["n_antennas"] < 2:
        return False, f"needs at least 2 elements, beamformer has {facts['n_antennas']}"
    if spec["uses_frequency_dimension"]:
        # Stated before the generic check below so a grouped bucket names
        # grouping rather than reporting an unreconstructable frequency set.
        if facts["grouped"]:
            return False, f"grouped feedback (Ng={facts['grouping_ng']}) defines no " \
                          "subcarrier positions"
        if facts["frequencies_hz"] is None:
            return False, "subcarrier frequencies could not be reconstructed"
    # MUSIC and ESPRIT need at least one noise eigenvector.
    if name in ("music", "esprit") and facts["n_antennas"] < 2:
        return False, "no noise subspace on a single element"
    return True, None


def solve_bucket(records, log_prefix, site, max_reports=None, gates=None):
    """
    Run every applicable estimator over one bucket, returning their results and
    the facts selection used. gates defaults to report_gates().
    """
    gates = report_gates() if gates is None else gates
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
                                         centre_mhz, head["nsubc"], head["ng"])

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
        "grouped": _is_grouped(head["standard"], head["ng"]),
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
        ok, reason = applicable(name, spec, facts, gates[name])
        if not ok:
            results.append({"estimator": name, "ran": False, "reason": reason})
            continue
        try:
            if spec["input"] == "covariance":
                out = spec["function"](covariance, positions, wavelength,
                                       n_snapshots=head["nsubc"],
                                       **spec.get("params", {}))
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


def solve(log_prefix, site_path=None, max_reports=None, out_path=None,
          tag=None):
    """
    Run Stage 3 over every bucket in the log and append the results.

    Returns (path, bucket count, timing). Timing separates reading the log from
    solving it: the read grows with the whole session's history, the solve with
    the bucket count and each estimator's grid.

    Results are appended, not overwritten. A later chunk adds reports to buckets
    already solved, so each pass is a fresh snapshot of every bucket rather than
    an increment; pass and tag identify which pass a record belongs to, and
    report() renders the last.

    lag_s is measured against wall clock, so it reads as pipeline lag on a live
    capture and as the recording's age on a replay.
    """
    started = time.perf_counter()
    gates = report_gates()
    site = load_site(site_path)

    mark = time.perf_counter()
    buckets = {}
    latest = None
    for record in observe.read_log(log_prefix):
        if record.get("kind") == "bfi":
            buckets.setdefault(record["bucket"], []).append(record)
            if latest is None or record["t"] > latest:
                latest = record["t"]
    read_s = time.perf_counter() - mark

    out_path = out_path or os.environ.get("SOLVE_OUT") or (log_prefix + ".solve.jsonl")
    pass_index = 0
    if os.path.exists(out_path):
        with open(out_path) as handle:
            for line in handle:
                if line.strip():
                    pass_index = max(pass_index,
                                     json.loads(line).get("pass", 0) + 1)
    written = 0
    per_estimator = {}
    mark = time.perf_counter()
    with open(out_path, "a") as handle:
        for key in sorted(buckets, key=lambda k: -len(buckets[k])):
            facts, results = solve_bucket(buckets[key], log_prefix, site,
                                          max_reports, gates)
            for result in results:
                if result.get("ran"):
                    per_estimator[result["estimator"]] = per_estimator.get(
                        result["estimator"], 0) + 1
            handle.write(json.dumps({"kind": "solve", "pass": pass_index,
                                     "tag": tag, "at": time.time(),
                                     "facts": facts, "results": results}) + "\n")
            written += 1
    solve_s = time.perf_counter() - mark

    timing = {
        "read_s": read_s,
        "solve_s": solve_s,
        "total_s": time.perf_counter() - started,
        "lag_s": (time.time() - latest) if latest is not None else None,
        "estimator_runs": per_estimator,
    }
    return out_path, written, timing


def report(solve_path, stream=sys.stdout):
    """
    Print the last solve pass as a table, readable without parsing JSON.

    Estimators are shown side by side rather than reconciled: agreement means
    the data supports a bearing, spread means it does not, and a single
    reconciled number would hide which.

    Bearings are relative to the assumed array. Against a nominal array they
    carry an unknown per-beamformer rotation, so what compares against a real
    room is the pattern across one beamformer's clients, not a single figure.
    """
    entries = [json.loads(line) for line in open(solve_path) if line.strip()]
    if not entries:
        return
    last = max(entry.get("pass", 0) for entry in entries)
    for entry in entries:
        if entry.get("pass", 0) != last:
            continue
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
        print("usage: dispatch.py <log_prefix> [site.json] [max_reports]\n"
              "  env: SOLVE_OUT, SOLVE_REPORT, TIMING_LOG, MIN_REPORTS_<ESTIMATOR>",
              file=sys.stderr)
        raise SystemExit(2)
    site_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else None
    path, count, timing = solve(sys.argv[1], site_arg, cap)
    lag = "" if timing["lag_s"] is None else f"  lag={timing['lag_s']:.2f}s"
    print(f"[stage3] {count} buckets -> {path}"
          f"   read={timing['read_s'] * 1e3:.0f}ms"
          f" solve={timing['solve_s'] * 1e3:.0f}ms"
          f" total={timing['total_s'] * 1e3:.0f}ms{lag}")
    timing_path = os.environ.get("TIMING_LOG")
    if timing_path:
        with open(timing_path, "a") as handle:
            handle.write(json.dumps({"stage": 3, "buckets": count,
                                     "timing": timing}) + "\n")
    if os.environ.get("SOLVE_REPORT"):
        with open(os.environ["SOLVE_REPORT"], "w") as handle:
            report(path, stream=handle)
    else:
        report(path)
