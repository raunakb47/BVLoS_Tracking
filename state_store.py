#!/usr/bin/env python3
"""
Module: state_store.py
Persistent per-bucket state shared across Stage 3 and Stage 4 invocations.

Every chunk runs in a freshly spawned subprocess (2_Stage2_Extraction.sh), so
nothing survives in memory between chunks. Four things downstream need to:
the sliding estimation window, the VSS-LMS filter state, the OS-CFAR reference
window, and track continuity for the UI.

State is a dict keyed by "{client_mac}_{ap_mac}_{pkt_config}", persisted with
the same np.save(allow_pickle=True) / .item() convention used elsewhere in the
pipeline.

Stage 3 and Stage 4 must be given SEPARATE state files (STATE_FILE and
TRACK_STATE_FILE). Stage 4 prunes buckets whose track has dropped; sharing one
file would delete Stage 3's estimation window for a bucket that is merely
quiet.
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
        "lms_floor_history": [],       # recent per-sample residual powers, floor estimate for the step-size rule
        "lms_floor_hold": 0,           # consecutive samples withheld from that history as suspected motion
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
    Advance track bookkeeping for every known bucket absent from the current
    chunk. Buckets present in the chunk are handled by mark_fix/mark_missed
    once KPVT/SSE has run.

    Returns the bucket_keys still worth rendering (CONFIRMED or COASTING), so
    Stage 4 can hold a position for a client that briefly stopped producing
    BFI, per the radar coast-then-drop pattern.
    """
    surviving = []
    for bucket_key, bucket_state in list(state.items()):
        if bucket_key in seen_bucket_keys:
            surviving.append(bucket_key)
            continue
        if bucket_state["last_seen_ts"] is None:
            # Never had an accepted fix (KPVT_ONLY bucket, or LOW SSE
            # confidence every chunk so far), so there is no position to coast.
            # Drop it rather than let mark_missed coast an unconfirmed track.
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
    trim to depth. Decouples how much history KPVT/SSE sees from the tcpdump
    rotation cadence: Stage 3 still runs once per chunk, but on up to
    window_chunks chunks of accumulated samples.
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
    Append this chunk's kinematic energy to the bucket's OS-CFAR reference
    history and trim to max_cells. Stores algorithm output, not input: it is
    the noise-floor reference 3_2_Kinematic_Tracker.py thresholds against.
    Every chunk is pushed unconditionally -- the order-statistic form does not
    require past detections to be excluded first.
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
    Advance a bucket's track by one missed chunk, following the radar
    confirmed/coasting/dropped pattern: coast across misses, delete after a
    bounded run of them rather than on a wall-clock timeout, which cannot tell
    "briefly quiet" from "gone". Returns False once the bucket should go.
    """
    bucket_state["coast_count"] += 1
    if bucket_state["coast_count"] > coast_limit:
        bucket_state["track_status"] = "DROPPED"
        return False
    bucket_state["track_status"] = "COASTING"
    return True
