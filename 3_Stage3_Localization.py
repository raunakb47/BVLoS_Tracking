#!/usr/bin/env python3
"""
Module: 3_Stage3_Localization.py
Evaluates matrix properties and routes them to the appropriate algorithm.

Each invocation of this script handles exactly one rotated pcap chunk and
then exits, so on its own it has no memory of the chunk before it. That used
to mean every KPVT/SSE estimate was computed from a single CHUNK_TIME-second
slice in isolation, with no continuity guarantee across the file-rotation
boundary. state_store.py gives each MAC/config bucket a small sliding window
of recent chunks (WINDOW_CHUNKS, persisted in STATE_FILE across invocations),
so the estimation window is decoupled from the tcpdump rotation cadence: a
bucket that goes quiet for part of one chunk still has its neighboring
chunks' samples available, and algorithms that need more than one chunk's
worth of packets to be stable (KPVT's variance estimate, SSE's covariance
estimate) get a wider, overlapping-in-effect base to work from.
"""
import sys
import os
import json
import numpy as np
from sklearn.decomposition import PCA
from importlib import import_module

spatial_algos = import_module('3_1_Spatial_Algorithms')
state_store = import_module('state_store')

KE_THRESHOLD = float(os.getenv("KE_THRESHOLD", 0.02))
STARVED_LIMIT = int(os.getenv("PACKET_STARVATION_LIMIT", 15))
WINDOW_CHUNKS = int(os.getenv("WINDOW_CHUNKS", state_store.DEFAULT_WINDOW_CHUNKS))

def kpvt_module(v_matrices):
    t_steps = v_matrices.shape[0]
    v_abs = np.abs(v_matrices).reshape(t_steps, -1)
    residual = v_abs - np.mean(v_abs, axis=0)
    variance_profile = PCA(n_components=1).fit_transform(residual).flatten()
    return float(np.var(variance_profile))

def dispatcher(sanitized_file, out_json, state_file=None):
    data = np.load(sanitized_file, allow_pickle=True).item()
    results = {}

    state = state_store.load_state(state_file) if state_file else {}

    for bucket_key, payload in data.items():
        client_mac, ap_mac, pkt_config = bucket_key.split('_')
        nt = int(pkt_config.split('x')[0])

        # Fold this chunk's samples into the bucket's sliding window before
        # running any estimator, so KPVT/SSE see up to WINDOW_CHUNKS chunks
        # of history instead of only the chunk that just arrived.
        if state_file:
            bucket_state = state_store.get_bucket(state, bucket_key)
            # 2_1_Temporal_Sanitizer.py resamples onto a fixed 100Hz grid and does not
            # retain per-sample wall-clock timestamps, so the window is trimmed by
            # chunk count (WINDOW_CHUNKS) rather than by sample age; the first tuple
            # slot is reserved for a future per-sample timestamp array if that changes.
            state_store.push_window_sample(
                bucket_state, None, payload['v_matrices'], payload['rssi'],
                window_chunks=WINDOW_CHUNKS
            )
            v_matrices = state_store.windowed_v_matrices(bucket_state)
            rssi_window = state_store.windowed_rssi(bucket_state)
        else:
            v_matrices = payload['v_matrices']
            rssi_window = payload['rssi']

        client_rssi = float(np.mean(rssi_window))

        ke = kpvt_module(v_matrices)
        packets = v_matrices.shape[0]

        ap_aod = None
        aod_confidence = None

        if nt < 2:
            routing, algo_name = "KPVT_ONLY", "NONE"
        else:
            routing = "KPVT_AND_SSE"
            if packets < STARVED_LIMIT: algo_name = "IAA_APES"
            elif ke < KE_THRESHOLD: algo_name = "SPOTFI"
            else: algo_name = "RES_2D_MUSIC" if nt >= 3 else "CA_ESPRIT"

            # Every SSE_REGISTRY entry reports its own confidence alongside the
            # angle: a HIGH-confidence estimate means the array's eigenvalue
            # spectrum actually looks like one dominant path (the assumption
            # the underlying math relies on); LOW means a second comparably
            # strong path was present, so the angle is likely a multipath
            # blend rather than the true AoD. See 3_1_Spatial_Algorithms.py.
            ap_aod, aod_confidence = spatial_algos.SSE_REGISTRY[algo_name](v_matrices, nt)

        results[bucket_key] = {
            "client_mac": client_mac,
            "ap_mac": ap_mac,
            "mimo": pkt_config,
            "routing": routing,
            "algorithm": algo_name,
            "kinematic_energy": ke,
            "ap_aod": ap_aod,
            "aod_confidence": aod_confidence,
            "client_rssi": client_rssi
        }

    if state_file:
        state_store.save_state(state_file, state)

    with open(out_json, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    state_arg = sys.argv[3] if len(sys.argv) > 3 else None
    dispatcher(sys.argv[1], sys.argv[2], state_arg)