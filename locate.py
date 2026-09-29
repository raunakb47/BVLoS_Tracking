#!/usr/bin/env python3
"""
Module: locate.py
Stage 4. Turn Stage 3 outputs into per-device fixes built from measured
quantities only, appended to a trajectory log.

What the data measures, and so what a fix may state:

  - The monitor has no bearing to anything; around it each transmitter is a
    range band [near_m, far_m] about median_m (ranging.py), any bound None
    where the path-loss model does not reach.
  - A bearing exists only in the beamforming AP's array frame, with its
    mirror about the array axis: theta or 180 - theta.
  - With an anchor (ANCHOR_MAC, the monitor host's own associated interface),
    the AP also measures its bearing to the monitor, phi. The angle at the AP
    from the monitor direction to a client is then measured, not assumed:
    delta in {theta - phi, 180 - theta - phi}. The whole scene may equally be
    its mirror image across the AP-monitor line (every delta negated): a
    linear array cannot tell the two apart, so both are kept.

No fix carries an orientation to the world or to the screen. A world frame
needs site input; none is assumed.

Output, one JSON record per line in <session>/stage4/locate.jsonl:

  kind "ap"   : an AP's range from the monitor, from a Stage 3 "ap" record
  kind "fix"  : one Stage 3 output of a client, with
                frame "anchor" (delta hypotheses) or "ap_array" (bearings),
                the client's range from the monitor, its AP's range
Each record names the Stage 3 line it came from (source_line), so a run
resumes after the last line it processed.
"""
import json
import os
import sys

import numpy as np

import ranging


def _wrap_deg(angle):
    """Angle in (-180, 180]."""
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def array_directions(bearing_deg):
    """The two directions in the AP array frame a linear-array bearing names."""
    return [bearing_deg, _wrap_deg(180.0 - bearing_deg)]


def relative_angles(client_deg, anchor_deg):
    """
    Angles at the AP from the monitor direction to the client, for one of the
    two mirror-image scenes; the other scene is every value negated.

    With the monitor direction taken as anchor_deg, the client is at
    client_deg or its mirror, giving these two. Taking the monitor at the
    anchor's mirror instead negates both, which is the scene's reflection.
    """
    return [_wrap_deg(client_deg - anchor_deg),
            _wrap_deg(180.0 - client_deg - anchor_deg)]


def _within(distance, band):
    """Whether distance lies in band's [near_m, far_m]; a None bound is open."""
    near, far = band.get("near_m"), band.get("far_m")
    return ((near is None or distance >= near - 1e-9)
            and (far is None or distance <= far + 1e-9))


def in_anchor_region(point, delta_deg, client_range, ap_range):
    """
    Whether a client position is consistent with one hypothesis, in the
    AP-centred frame whose reference ray points at the monitor.

    point is (s, r): the client's distance from the AP along delta, and the
    monitor's distance from the AP along the reference ray. Consistent when
    r is within the AP's range band and the client's distance from the
    monitor within its own. A missing range constrains nothing.
    """
    s, r = point
    if s < 0 or r <= 0:
        return False
    if ap_range and not _within(r, ap_range):
        return False
    delta = np.radians(delta_deg)
    gap = float(np.hypot(s * np.cos(delta) - r, s * np.sin(delta)))
    return not client_range or _within(gap, client_range)


def _latest_anchor(anchor_outputs, beamformer, estimator):
    """
    The newest primary output of the anchor towards beamformer, from the same
    estimator when it has one, else from any; None before the first.
    """
    outputs = anchor_outputs.get(beamformer) or []
    same = [o for o in outputs if o["estimator"] == estimator]
    return (same or outputs or [None])[-1]


def fixes(solve_path, start_line, anchor_mac, regdomain, model_name, k):
    """
    Yield Stage 4 records for Stage 3 lines from start_line on.

    Earlier lines are still read for context: AP records bound every AP's
    power and range, and anchor outputs set the reference direction.
    """
    ap_records, ap_ranges, anchor_outputs = [], {}, {}
    anchor_mac = (anchor_mac or "").lower() or None
    with open(solve_path) as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            entry = json.loads(line)
            new = line_number >= start_line

            if entry.get("kind") == "ap":
                ap_records.append(entry)
                band = ranging.ap_range(entry, ap_records, regdomain, model_name, k)
                for beamformer in entry.get("beamformers") or [entry["bssid"]]:
                    ap_ranges[beamformer] = dict(band, t_last=entry["t_last"],
                                                 bssid=entry["bssid"])
                if new:
                    yield {"kind": "ap", "source_line": line_number,
                           "bssid": entry["bssid"],
                           "beamformers": entry.get("beamformers", []),
                           "t_first": entry["t_first"], "t_last": entry["t_last"],
                           "range": band}
                continue

            facts = entry["facts"]
            primaries = [r for r in entry["results"]
                         if r.get("ran") and r.get("role", "primary") == "primary"
                         and r["candidates"]]
            if anchor_mac and facts["transmitter"] == anchor_mac:
                for output in primaries:
                    anchor_outputs.setdefault(facts["beamformer"], []).append(
                        dict(output, source_line=line_number))
                continue
            if not new:
                continue

            ap_range = ap_ranges.get(facts["beamformer"])
            for output in primaries:
                anchor = (_latest_anchor(anchor_outputs, facts["beamformer"],
                                         output["estimator"]) if anchor_mac else None)
                record = {
                    "kind": "fix",
                    "source_line": line_number,
                    "device": facts["transmitter"],
                    "beamformer": facts["beamformer"],
                    "bucket": facts["bucket"],
                    "estimator": output["estimator"],
                    "slice_id": output["slice_id"],
                    "t_first": output["t_first"],
                    "t_last": output["t_last"],
                    "n_reports": output["n_reports"],
                    "geometry_source": facts["geometry_source"],
                    "client_range": ranging.client_range(output, facts, ap_records,
                                                         regdomain, model_name, k),
                    "ap_range": ap_range,
                }
                bearings = [(c["bearing_deg"], c["strength"]) for c in output["candidates"]]
                if anchor is None:
                    record["frame"] = "ap_array"
                    record["hypotheses"] = [
                        {"directions_deg": array_directions(b), "strength": s}
                        for b, s in bearings]
                else:
                    # Every anchor candidate is kept: its strongest peak is
                    # not known to be the path to the monitor.
                    record["frame"] = "anchor"
                    record["anchor"] = {"slice_id": anchor["slice_id"],
                                        "estimator": anchor["estimator"],
                                        "t_last": anchor["t_last"]}
                    record["hypotheses"] = [
                        {"bearing_deg": b, "anchor_bearing_deg": c["bearing_deg"],
                         "delta_deg": relative_angles(b, c["bearing_deg"]),
                         "strength": s, "anchor_strength": c["strength"]}
                        for b, s in bearings for c in anchor["candidates"]]
                yield record


def locate(solve_path, session_dir, environ=None):
    """
    Append fixes for Stage 3 lines not yet processed; returns the count.
    """
    environ = os.environ if environ is None else environ
    model_name, _ = ranging.model_from_env(environ)
    k = ranging.sigma_k(environ)
    regdomain = ranging.load_regdomain(os.path.join(session_dir, "regdomain.txt"))
    out_dir = os.path.join(session_dir, "stage4")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "locate.jsonl")

    start_line = 0
    if os.path.exists(out_path):
        with open(out_path) as handle:
            for line in handle:
                if line.strip():
                    start_line = max(start_line, json.loads(line)["source_line"] + 1)

    written = 0
    with open(out_path, "a") as handle:
        for record in fixes(solve_path, start_line, environ.get("ANCHOR_MAC"),
                            regdomain, model_name, k):
            handle.write(json.dumps(record) + "\n")
            written += 1
    return out_path, written


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: locate.py <solve.jsonl> <session_dir>\n"
              "  env: ANCHOR_MAC, PATH_LOSS_MODEL, RANGE_SIGMA_K", file=sys.stderr)
        raise SystemExit(2)
    path, count = locate(sys.argv[1], sys.argv[2])
    print(f"[stage4] {count} records -> {path}")
