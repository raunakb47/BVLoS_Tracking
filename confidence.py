#!/usr/bin/env python3
"""
Module: confidence.py
Stage 4 confidence reconciliation. Grades each segment of a device's track by
how many independent estimator groups back one direction, and revises it once
when a longer slice covering it arrives.

Segment. The smallest slice Stage 3 output for a bucket: under carry-until-
gate, the gate-1 estimators' slice of each pass, all of them on identical
reports. A later slice that contains segments whole (MUSIC's, gate 10, with
its companions on the same reports) is a covering slice: each segment inside
it is revised once, taking the covering slice's bearing and step only when
that confidence is higher, else kept. Segments not yet covered stay
provisional.

Correlation groups, from how aoa.py computes each estimator:
  subspace        music, esprit    eigendecomposition of the covariance
  covariance_fit  spice            convex covariance fit, start-independent
  weighted_ls     samv2, iaa       same periodogram start, same R^-1 a update
Agreement within a group is not independent corroboration, so support is
counted per group. All groups still read one covariance.

Settings, from the environment (config.env), defaults in DEFAULTS:
  CONF_GROUPS            correlation groups, "name=est,est;name=est"; an
                         estimator not listed is its own group
  CONF_TOLERANCE_FACTOR  cluster tolerance in u = sin(theta), as a multiple of
                         lambda / (N d): 0.886 is the half-power beamwidth of a
                         uniform aperture, so directions closer than this are
                         not resolved by the array. Non-ULA or unknown spacing:
                         undetermined, and the segment is left ungraded
  CONF_SUPPORT_DB        a candidate supports a direction within this many dB
                         of its own estimator's strongest (3: half power).
                         Strengths are never compared across estimators
  CONF_REVISION          what a covering slice of higher confidence does:
                         covering     segment takes its bearing and step
                         corroborate  segment keeps its own bearing(s) that lie
                                      within tolerance of the covering leader
                                      and takes the step; covering bearing only
                                      when none does
                         off          no revision: every covered segment kept

Step = number of groups supporting the leading cluster (1, 2 or 3).
Dotted = another cluster ties the leader, so no one direction is backed.

Output, one JSON record per segment state in <session>/stage4/segments.jsonl:
  state "provisional"  when the segment's slice arrives
  state "final"        instead, when every live estimator already ran on
                       that slice, so no covering slice will follow
  state "revised"      the covering slice's confidence was higher
  state "kept"         it was not
Client segments carry trajectory points in the monitor frame when an anchor
is set (locate.anchor_points), from the leading bearing(s) and the anchor's.
"""
import json
import os
import sys

import numpy as np

import locate
import ranging

DEFAULTS = {
    "CONF_GROUPS": "subspace=music,esprit;covariance_fit=spice;weighted_ls=samv2,iaa",
    "CONF_TOLERANCE_FACTOR": "0.886",
    "CONF_SUPPORT_DB": "3",
    "CONF_REVISION": "covering",
}
REVISION_MODES = ("covering", "corroborate", "off")
_C_LIGHT = 299_792_458.0


def parse_groups(text):
    """{estimator: group} from "name=est,est;name=est"."""
    groups = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        name, _, members = part.partition("=")
        if not name.strip() or not members.strip():
            raise ValueError(f"CONF_GROUPS: expected name=estimator,... in {part!r}")
        for estimator in filter(None, (m.strip() for m in members.split(","))):
            if estimator in groups:
                raise ValueError(f"CONF_GROUPS: {estimator} is in two groups")
            groups[estimator] = name.strip()
    return groups


def settings_from_env(environ):
    """The reconciliation settings, each from the environment or DEFAULTS."""
    get = lambda key: (environ.get(key) or DEFAULTS[key]).strip()
    revision = get("CONF_REVISION")
    if revision not in REVISION_MODES:
        raise ValueError(f"CONF_REVISION must be one of {REVISION_MODES}, not {revision!r}")
    factor, support_db = float(get("CONF_TOLERANCE_FACTOR")), float(get("CONF_SUPPORT_DB"))
    if factor <= 0 or support_db < 0:
        raise ValueError("CONF_TOLERANCE_FACTOR must be > 0 and CONF_SUPPORT_DB >= 0")
    return {"groups": parse_groups(get("CONF_GROUPS")), "tolerance_factor": factor,
            "support_db": support_db, "revision": revision}


SETTINGS = settings_from_env({})


def group_of(estimator, settings=SETTINGS):
    """The correlation group of an estimator; an unlisted one is its own."""
    return settings["groups"].get(estimator, estimator)


def tolerance_u(facts, settings=SETTINGS):
    """
    Half-power beamwidth in u = sin(theta) of the beamformer's array, or None
    where the geometry does not give one.

    The spacing is Stage 3's element_spacing_m; a Stage 3 log without it
    falls back to the nominal half-wavelength array only when that is the
    geometry it records having used.
    """
    n, centre = facts.get("n_antennas"), facts.get("centre_freq_mhz")
    if not n or n < 2 or not centre:
        return None
    wavelength = _C_LIGHT / (centre * 1e6)
    if "element_spacing_m" in facts:
        spacing = facts["element_spacing_m"]
    elif facts.get("geometry_source") == "nominal_half_wavelength_ula":
        spacing = wavelength / 2.0
    else:
        spacing = None
    if not spacing:
        return None
    return settings["tolerance_factor"] * wavelength / (n * spacing)


def supporting(output, settings=SETTINGS):
    """An output's candidates within support_db of its own strongest."""
    candidates = output.get("candidates") or []
    if not candidates:
        return []
    floor = max(c["strength"] for c in candidates) * 10.0 ** (-settings["support_db"] / 10.0)
    return [c for c in candidates if c["strength"] >= floor]


def clusters(outputs, tolerance, settings=SETTINGS):
    """
    Supporting candidates of every output, clustered by single linkage in
    u = sin(theta) at tolerance. Only supporting candidates are clustered, so
    a weak candidate cannot link two strong directions. Strongest first:
    most groups, then most estimators.
    """
    points = sorted((float(np.sin(np.radians(c["bearing_deg"]))), c["bearing_deg"],
                     output["estimator"])
                    for output in outputs for c in supporting(output, settings))
    found, current = [], []
    for point in points:
        if current and point[0] - current[-1][0] > tolerance:
            found.append(current)
            current = []
        current.append(point)
    if current:
        found.append(current)
    summary = []
    for members in found:
        estimators = sorted({m[2] for m in members})
        summary.append({"bearing_deg": float(np.median([m[1] for m in members])),
                        "estimators": estimators,
                        "groups": sorted({group_of(e, settings) for e in estimators})})
    summary.sort(key=lambda c: (-len(c["groups"]), -len(c["estimators"])))
    return summary


def assess(outputs, tolerance, settings=SETTINGS):
    """
    Step, dotted and the leading bearing(s) of one slice's outputs. With no
    tolerance the slice is ungraded: step None.
    """
    estimators = sorted(o["estimator"] for o in outputs)
    base = {"estimators": estimators,
            "groups_present": sorted({group_of(e, settings) for e in estimators}),
            "tolerance_u": tolerance}
    if tolerance is None:
        return dict(base, step=None, dotted=None, bearings_deg=[], clusters=[],
                    reason="beamwidth undetermined: not a uniform linear array of known spacing")
    found = clusters(outputs, tolerance, settings)
    if not found:
        return dict(base, step=None, dotted=None, bearings_deg=[], clusters=[],
                    reason="no candidates")
    step = len(found[0]["groups"])
    leaders = [c for c in found if len(c["groups"]) == step]
    return dict(base, step=step, dotted=len(leaders) > 1,
                bearings_deg=[c["bearing_deg"] for c in leaders], clusters=found)


def _rank(assessment):
    """Order of confidence: more groups first, then undotted over dotted."""
    if assessment["step"] is None:
        return (-1, False)
    return (assessment["step"], not assessment["dotted"])


def revise(own, covering, mode):
    """
    (assessment, state) for a segment whose slice a covering slice contains.
    Only a covering slice of higher confidence changes anything.
    """
    if mode == "off" or _rank(covering) <= _rank(own):
        return own, "kept"
    if mode == "corroborate" and covering["tolerance_u"] is not None:
        u = lambda b: np.sin(np.radians(b))
        corroborated = [b for b in own["bearings_deg"]
                        if any(abs(u(b) - u(c)) <= covering["tolerance_u"]
                               for c in covering["bearings_deg"])]
        if corroborated:
            return dict(covering, bearings_deg=corroborated, bearings_from="segment"), "revised"
    return dict(covering, bearings_from="covering"), "revised"


def _live(entry):
    """Estimators live for this bucket: every result not refused as absent from the live set."""
    return {r["estimator"] for r in entry["results"]
            if r.get("ran") or not str(r.get("reason", "")).startswith("not in ")}


def _slices(entry):
    """A Stage 3 line's outputs grouped by slice, shortest slice first."""
    by_slice = {}
    for result in entry["results"]:
        if result.get("ran") and result.get("candidates"):
            by_slice.setdefault(result["slice_id"], []).append(result)
    return sorted(by_slice.values(), key=lambda outs: (outs[0]["n_reports"], outs[0]["t_last"]))


def _points(assessment, anchor, ap_range, client_range):
    """Trajectory candidates for a client segment against the anchor's bearings."""
    if not anchor or not anchor["bearings_deg"]:
        return []
    hypotheses = [{"bearing_deg": b, "anchor_bearing_deg": phi,
                   "delta_deg": locate.relative_angles(b, phi)}
                  for b in assessment["bearings_deg"] for phi in anchor["bearings_deg"]]
    return locate.anchor_points(hypotheses, ap_range, client_range)


def segments(solve_path, start_line, anchor_mac, regdomain, model_name, k,
             settings=SETTINGS):
    """
    Yield segment records for Stage 3 lines from start_line on. Earlier lines
    are read again for state: open segments, AP ranges, anchor bearings.
    """
    ap_records, ap_ranges = [], {}
    anchors = {}           # beamformer -> latest anchor segment's current assessment
    open_segments = {}     # bucket -> [segment state] not yet covered
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
                continue

            facts = entry["facts"]
            is_anchor = bool(anchor_mac) and facts["transmitter"] == anchor_mac
            tolerance = tolerance_u(facts, settings)
            pending = open_segments.setdefault(facts["bucket"], [])

            def emit(segment, state, covering=None):
                assessment = segment["assessment"]
                record = dict(segment["base"], state=state, source_line=line_number,
                              **{k_: assessment[k_] for k_ in
                                 ("step", "dotted", "bearings_deg", "clusters", "estimators",
                                  "groups_present", "tolerance_u")})
                for key in ("reason", "bearings_from"):
                    if key in assessment:
                        record[key] = assessment[key]
                record["assessed_on"] = segment["assessed_on"]
                if covering is not None:
                    record["covering"] = {"slice_id": covering["slice_id"],
                                          "step": covering["step"],
                                          "dotted": covering["dotted"]}
                if is_anchor:
                    record["role"] = "anchor"
                    record["frame"] = "ap_array"
                else:
                    anchor = anchors.get(facts["beamformer"]) if anchor_mac else None
                    ap_range = ap_ranges.get(facts["beamformer"])
                    record["role"] = "client"
                    record["ap_range"] = ap_range
                    if anchor is None:
                        record["frame"] = "ap_array"
                    else:
                        record["frame"] = "anchor"
                        record["anchor"] = anchor
                        record["points"] = _points(assessment, anchor, ap_range,
                                                   record["client_range"])
                latest = anchors.get(facts["beamformer"])
                if is_anchor and (latest is None
                                  or segment["base"]["t_last"] >= latest["t_last"]):
                    anchors[facts["beamformer"]] = {
                        "slice_id": segment["base"]["slice_id"],
                        "t_last": segment["base"]["t_last"], "step": assessment["step"],
                        "dotted": assessment["dotted"], "bearings_deg": assessment["bearings_deg"]}
                return record

            for outputs in _slices(entry):
                head = outputs[0]
                slice_ = {"slice_id": head["slice_id"], "t_first": head["t_first"],
                          "t_last": head["t_last"]}
                assessment = assess(outputs, tolerance, settings)
                inside = [s for s in pending
                          if s["base"]["slice_id"] != slice_["slice_id"]
                          and slice_["t_first"] <= s["base"]["t_first"]
                          and s["base"]["t_last"] <= slice_["t_last"]]
                if inside:
                    covering = dict(slice_, step=assessment["step"], dotted=assessment["dotted"])
                    for segment in inside:
                        pending.remove(segment)
                        revised, state = revise(segment["assessment"], assessment,
                                                settings["revision"])
                        if state == "revised":
                            segment["assessment"] = revised
                            segment["assessed_on"] = slice_["slice_id"]
                        record = emit(segment, state, covering)
                        if new:
                            yield record
                    continue

                base = {"kind": "segment", "device": facts["transmitter"],
                        "beamformer": facts["beamformer"], "bucket": facts["bucket"],
                        "slice_id": head["slice_id"], "t_first": head["t_first"],
                        "t_last": head["t_last"], "n_reports": head["n_reports"]}
                if not is_anchor:
                    base["client_range"] = ranging.client_range(head, facts, ap_records,
                                                                regdomain, model_name, k)
                segment = {"base": base, "assessment": assessment,
                           "assessed_on": head["slice_id"]}
                if {o["estimator"] for o in outputs} >= _live(entry):
                    record = emit(segment, "final")
                else:
                    pending.append(segment)
                    record = emit(segment, "provisional")
                if new:
                    yield record


def reconcile(solve_path, session_dir, environ=None):
    """Append segment records for Stage 3 lines not yet processed; returns the count."""
    environ = os.environ if environ is None else environ
    model_name, _ = ranging.model_from_env(environ)
    k = ranging.sigma_k(environ)
    settings = settings_from_env(environ)
    regdomain = ranging.load_regdomain(os.path.join(session_dir, "regdomain.txt"))
    out_dir = os.path.join(session_dir, "stage4")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "segments.jsonl")

    start_line = 0
    if os.path.exists(out_path):
        with open(out_path) as handle:
            for line in handle:
                if line.strip():
                    start_line = max(start_line, json.loads(line)["source_line"] + 1)

    written = 0
    with open(out_path, "a") as handle:
        for record in segments(solve_path, start_line, environ.get("ANCHOR_MAC"),
                               regdomain, model_name, k, settings):
            handle.write(json.dumps(record) + "\n")
            written += 1
    return out_path, written


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: confidence.py <solve.jsonl> <session_dir>\n"
              "  env: ANCHOR_MAC, PATH_LOSS_MODEL, RANGE_SIGMA_K, CONF_GROUPS,\n"
              "       CONF_TOLERANCE_FACTOR, CONF_SUPPORT_DB, CONF_REVISION", file=sys.stderr)
        raise SystemExit(2)
    path, count = reconcile(sys.argv[1], sys.argv[2])
    print(f"[stage4] {count} segment records -> {path}")
