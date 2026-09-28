#!/usr/bin/env python3
"""
Module: bench_precision.py
Estimator precision on captured data.

No capture carries a labelled client position, so nothing here measures
accuracy. Every number is precision: whether an estimator returns the same
answer when asked twice from different data. That is the right question for a
method claiming to work when the covariance is poorly conditioned, since such a
claim is that its answer stops moving sooner as reports accumulate. An estimator
converging quickly to a wrong bearing would still score well, so this narrows
what a capture with known geometry has left to settle rather than replacing it.

bench_aoa.py is the complement: synthetic, with truth, for correctness.

Two measures:

  Subsample convergence -- each estimator's all-reports answer is its own
  reference; the estimate is recomputed from random subsets and scored against
  it. Each estimator is compared only against itself, since comparing across
  them would fold in their disagreement about the bearing.

  Split-half repeatability -- two disjoint halves solved independently and
  compared.

Differences are folded into [0, 90]: a linear array's response depends on
sin(bearing), so a bearing and its mirror about the array axis are
indistinguishable and scoring them apart would penalise a distinction the
geometry does not support.

Run: python3 bench_precision.py <log_prefix> [log_prefix ...]
"""
import sys

import numpy as np

import aoa
import observe

SUBSET_SIZES = (1, 2, 5, 10, 20, 50)
DRAWS = 40
# Reports drawn from a bucket before benchmarking. The sweep only asks about
# subsets up to 50, so a few hundred is well past where the curve flattens.
# Drawn uniformly across the bucket's span rather than from one moment.
REFERENCE_CAP = 300
_C_LIGHT = 299792458.0


def circular_error_deg(a_deg, b_deg):
    """
    Separation between two bearings from a linear array, in [0, 90].

    Folded because a bearing and its mirror about the array axis produce
    identical steering vectors.
    """
    diff = abs(float(a_deg) - float(b_deg)) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def _top_bearing(result):
    if not result["candidates_rad"]:
        return None
    return float(np.degrees(result["candidates_rad"][0][0]))


def _estimate(name, v_stack, gains_db, positions, wavelength, frequencies,
              n_subcarriers):
    """One estimator's leading bearing from a set of reports, or None."""
    spec = aoa.ESTIMATORS[name]
    try:
        if spec["input"] == "covariance":
            covariance = aoa.bff_covariance(v_stack, gains_db)
            out = spec["function"](covariance, positions, wavelength,
                                   n_snapshots=n_subcarriers,
                                   **spec.get("params", {}))
        else:
            if frequencies is None:
                return None
            out = spec["function"](v_stack, gains_db, positions, frequencies)
    except Exception:                                     # noqa: BLE001
        return None
    return _top_bearing(out)


def bucket_precision(records, log_prefix, estimators, seed=0):
    """Both measures, for one bucket, for each estimator."""
    import dispatch

    rng = np.random.default_rng(seed)
    records = sorted(records, key=lambda r: r["t"])
    if len(records) > REFERENCE_CAP:
        keep = np.linspace(0, len(records) - 1, REFERENCE_CAP).astype(int)
        records = [records[i] for i in keep]
    head = records[0]
    centre_mhz = head.get("freq_mhz")
    if centre_mhz is None:
        return None
    wavelength = _C_LIGHT / (centre_mhz * 1e6)
    positions = aoa.uniform_linear(head["nr"], wavelength / 2.0)
    frequencies = dispatch.subcarrier_frequencies(
        head["standard"], head["bw"], centre_mhz, head["nsubc"], head["ng"])

    v_stack = np.stack([observe.read_v_matrix(r, log_prefix) for r in records])
    gains_db = np.array([r["stream_snr"] for r in records], dtype=float)
    n = len(records)

    out = {}
    for name in estimators:
        reference = _estimate(name, v_stack, gains_db, positions, wavelength,
                              frequencies, head["nsubc"])
        if reference is None:
            continue
        sweep = {}
        for size in SUBSET_SIZES:
            if size > n:
                continue
            errors = []
            for _ in range(DRAWS):
                pick = rng.choice(n, size=size, replace=False)
                got = _estimate(name, v_stack[pick], gains_db[pick], positions,
                                wavelength, frequencies, head["nsubc"])
                if got is not None:
                    errors.append(circular_error_deg(got, reference))
            if errors:
                sweep[size] = (float(np.median(errors)),
                               float(np.percentile(errors, 90)))

        halves = None
        if n >= 4:
            order = rng.permutation(n)
            a, b = order[:n // 2], order[n // 2:]
            ea = _estimate(name, v_stack[a], gains_db[a], positions, wavelength,
                           frequencies, head["nsubc"])
            eb = _estimate(name, v_stack[b], gains_db[b], positions, wavelength,
                           frequencies, head["nsubc"])
            if ea is not None and eb is not None:
                halves = circular_error_deg(ea, eb)

        out[name] = {"reference_deg": reference, "sweep": sweep,
                     "split_half_deg": halves}
    return out


def run(log_prefixes, estimators=None, min_reports=40):
    estimators = estimators or list(aoa.ESTIMATORS)
    collected = {name: {size: [] for size in SUBSET_SIZES} for name in estimators}
    splits = {name: [] for name in estimators}
    n_buckets = 0

    for prefix in log_prefixes:
        buckets = {}
        for record in observe.read_log(prefix):
            if record.get("kind") == "bfi":
                buckets.setdefault(record["bucket"], []).append(record)
        for key, records in sorted(buckets.items()):
            if len(records) < min_reports:
                continue
            head = records[0]
            result = bucket_precision(records, prefix, estimators)
            if not result:
                continue
            n_buckets += 1
            print(f"\n  {key}   {head['nr']}x{head['nc']} @ {head['bw']} MHz "
                  f"{head['standard']}/{head['feedback']}   {len(records)} reports")
            for name, data in result.items():
                row = "  ".join(
                    f"n={size}:{data['sweep'][size][0]:5.1f}"
                    for size in SUBSET_SIZES if size in data["sweep"])
                split = ("  split-half " + f"{data['split_half_deg']:5.1f}"
                         if data["split_half_deg"] is not None else "")
                print(f"     {name:17s} ref {data['reference_deg']:+7.1f}   {row}{split}")
                for size, (median, _) in data["sweep"].items():
                    collected[name][size].append(median)
                if data["split_half_deg"] is not None:
                    splits[name].append(data["split_half_deg"])

    print(f"\n{'=' * 78}")
    print(f"AGGREGATE over {n_buckets} buckets -- median deviation from each")
    print("estimator's own all-reports answer, in degrees (lower is steadier)")
    print(f"{'=' * 78}")
    header = "  ".join(f"n={s:<4}" for s in SUBSET_SIZES)
    print(f"  {'estimator':17s} {header}   split-half")
    for name in estimators:
        cells = []
        for size in SUBSET_SIZES:
            values = collected[name][size]
            cells.append(f"{np.median(values):6.1f}" if values else "     -")
        split = splits[name]
        tail = f"{np.median(split):9.1f}" if split else "        -"
        print(f"  {name:17s} " + "  ".join(cells) + tail)
    print("\n  Precision only: no capture here carries a known client position.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: bench_precision.py <log_prefix> [log_prefix ...]",
              file=sys.stderr)
        raise SystemExit(2)
    sys.exit(run(sys.argv[1:]))
