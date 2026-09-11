#!/usr/bin/env python3
"""
Module: state_store.py
Persistent per-bucket state shared across Stage 3 and Stage 4 invocations.

Each pcap chunk is processed by a freshly spawned Stage 3 / Stage 4 subprocess
(see 2_Stage2_Extraction.sh), so nothing survives in process memory between
chunks. Several techniques used downstream need memory that outlives a single
chunk: a sliding estimation window (accumulating more than one chunk's samples
per bucket before running KPVT/SSE), an adaptive filter's internal state
(VSS-LMS background estimate and step size), a rolling reference window for
order-statistic detection (OS-CFAR), and track continuity for the UI (holding
a bucket's last known position, and its position's age, across chunks where
that bucket produced no fresh packets at all).

This module is the on-disk handoff for that memory: a dict keyed by
bucket_key ("{client_mac}_{ap_mac}_{pkt_config}"), persisted with the same
np.save(..., allow_pickle=True) / .item() convention already used elsewhere
in this pipeline for dict-of-array payloads, so it round-trips through the
same tooling without introducing a second serialization format.
"""
import os
import time
import numpy as np

DEFAULT_WINDOW_CHUNKS = 3      # sliding-window depth, in chunks, retained per bucket
DEFAULT_COAST_LIMIT = 3        # consecutive chunks a bucket may go unseen before its track is dropped


def load_state(state_path):
    """Load the persistent state dict, or an empty one on first run."""
    if not os.path.exists(state_path):
        return {}
    return np.load(state_path, allow_pickle=True).item()


def save_state(state_path, state):
    """Persist the state dict. Called once per Stage 3 invocation, after processing."""
    np.save(state_path, state)


def _new_bucket_state():
    return {
        "window": [],                  # list of (timestamps, v_matrices, rssi) tuples, oldest first, one per chunk
        "lms_background": None,        # VSS-LMS complex background estimate, shape matches one v_matrix entry
        "lms_mu": None,                # VSS-LMS current per-bucket step size
        "cfar_reference": [],          # rolling kinematic-energy reference window consumed by OS-CFAR
        "last_position": None,         # [x, y] of the most recent accepted SSE fix
        "last_ap_aod": None,           # AoD (degrees) backing that fix, retained for the UI vector display
        "last_seen_ts": None,          # time.time() timestamp of that fix, used to compute staleness/age
        "coast_count": 0,              # consecutive chunks since the last accepted fix
        "track_status": "TENTATIVE",   # TENTATIVE -> CONFIRMED -> COASTING -> DROPPED
    }


def get_bucket(state, bucket_key):
    """Fetch a bucket's state, creating it on first encounter."""
    return state.setdefault(bucket_key, _new_bucket_state())


def touch_buckets(state, seen_bucket_keys, coast_limit=DEFAULT_COAST_LIMIT):
    """
    Advance track-management bookkeeping for every previously known bucket that
    did NOT appear in the current chunk at all (zero packets that interval).
    Buckets seen in the current chunk are left untouched here; their state is
    updated later via mark_fix()/mark_missed() once KPVT/SSE has run.

    Returns the list of bucket_keys whose track should still be rendered
    (CONFIRMED or COASTING), so Stage 4 can paint held-over positions for
    clients that briefly stopped producing BFI packets, per the standard
    radar coast-then-drop pattern (predict/hold across missed detections,
    drop the track only after a bounded run of consecutive misses).
    """
    surviving = []
    for bucket_key, bucket_state in list(state.items()):
        if bucket_key in seen_bucket_keys:
            surviving.append(bucket_key)
            continue
        if bucket_state["last_seen_ts"] is None:
            # This entry exists only because Stage 3 created it to track a
            # sliding window (see push_window_sample); it has never had an
            # accepted position fix (mark_fix was never called for it -- e.g.
            # a KPVT_ONLY bucket, or one whose SSE confidence has been LOW
            # every chunk so far). There is nothing to coast/hold, so it is
            # dropped from state rather than fed into mark_missed(), which
            # would otherwise "coast" a track that was never actually confirmed.
            del state[bucket_key]
            continue
        still_alive = mark_missed(bucket_state, coast_limit)
        if still_alive:
            surviving.append(bucket_key)
        else:
            del state[bucket_key]
    return surviving


def push_window_sample(bucket_state, timestamps, v_matrices, rssi, window_chunks=DEFAULT_WINDOW_CHUNKS):
    """
    Append one chunk's sanitized samples to the bucket's rolling window and
    trim to the configured depth. This decouples the estimation window (how
    much history KPVT/SSE sees) from both the tcpdump rotation cadence and
    the output stride (Stage 3 still runs once per chunk arrival, but each
    run now sees up to window_chunks chunks of accumulated history instead
    of only the newest one).
    """
    bucket_state["window"].append((timestamps, v_matrices, rssi))
    if len(bucket_state["window"]) > window_chunks:
        bucket_state["window"] = bucket_state["window"][-window_chunks:]


def windowed_v_matrices(bucket_state):
    return np.concatenate([w[1] for w in bucket_state["window"]], axis=0)


def windowed_rssi(bucket_state):
    return np.concatenate([w[2] for w in bucket_state["window"]], axis=0)


def push_cfar_reference(bucket_state, kinematic_energy, max_cells):
    """
    Append this chunk's kinematic-energy value to the bucket's OS-CFAR
    reference history and trim to max_cells. Unlike push_window_sample (which
    feeds algorithm inputs), this stores algorithm OUTPUT: the running record
    of "what did this bucket's kinematic energy look like recently" that
    3_2_Kinematic_Tracker.py's OS-CFAR detector treats as its noise-floor
    reference window (see Rohling 1983 -- the order-statistic form
    deliberately does not need this history to already exclude past
    detections, which is why every chunk's value is pushed unconditionally).
    """
    bucket_state["cfar_reference"].append(float(kinematic_energy))
    if len(bucket_state["cfar_reference"]) > max_cells:
        bucket_state["cfar_reference"] = bucket_state["cfar_reference"][-max_cells:]


def mark_fix(bucket_state, position, ap_aod):
    """Record a fresh, accepted SSE position fix and reset track staleness."""
    bucket_state["last_position"] = [float(position[0]), float(position[1])]
    bucket_state["last_ap_aod"] = float(ap_aod) if ap_aod is not None else None
    bucket_state["last_seen_ts"] = time.time()
    bucket_state["coast_count"] = 0
    bucket_state["track_status"] = "CONFIRMED"


def mark_missed(bucket_state, coast_limit=DEFAULT_COAST_LIMIT):
    """
    Advance a bucket's track by one missed chunk. Mirrors the confirmed /
    coasting / dropped pattern standard in radar target tracking (coast
    across missed detections, delete only after a bounded run of misses,
    rather than a raw wall-clock timeout that can't distinguish "briefly
    quiet" from "gone"). Returns False once the bucket should be removed
    from the live display entirely.
    """
    bucket_state["coast_count"] += 1
    if bucket_state["coast_count"] > coast_limit:
        bucket_state["track_status"] = "DROPPED"
        return False
    bucket_state["track_status"] = "COASTING"
    return True
