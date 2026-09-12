#!/usr/bin/env python3
"""
Module: 3_Stage3_Localization.py
Route each bucket to a KPVT/SSE estimator based on its array size, packet
count and kinematic energy, and write the per-bucket result for Stage 4.

One invocation handles one rotated pcap chunk and exits, so continuity comes
from STATE_FILE (state_store.py): each bucket carries a sliding window of the
last WINDOW_CHUNKS chunks, so KPVT's variance estimate and SSE's covariance
estimate are not limited to whatever landed inside one rotation boundary.
"""
import sys
import os
import json
import numpy as np
from importlib import import_module

spatial_algos = import_module('3_1_Spatial_Algorithms')
state_store = import_module('state_store')
kinematic_tracker = import_module('3_2_Kinematic_Tracker')
tx_power_estimator = import_module('tx_power_estimator')

KE_THRESHOLD = float(os.getenv("KE_THRESHOLD", 0.02))
STARVED_LIMIT = int(os.getenv("PACKET_STARVATION_LIMIT", 15))
WINDOW_CHUNKS = int(os.getenv("WINDOW_CHUNKS", state_store.DEFAULT_WINDOW_CHUNKS))
CFAR_REFERENCE_CELLS = int(os.getenv("CFAR_REFERENCE_CELLS", 32))
CFAR_PFA = float(os.getenv("CFAR_PFA", 1e-3))

def dispatcher(sanitized_file, out_json, state_file=None):
    data = np.load(sanitized_file, allow_pickle=True).item()
    results = {}

    state = state_store.load_state(state_file) if state_file else {}

    for bucket_key, payload in data.items():
        client_mac, ap_mac, pkt_config = bucket_key.split('_')
        nt = int(pkt_config.split('x')[0])

        bucket_state = state_store.get_bucket(state, bucket_key) if state_file else None

        # Fold this chunk into the sliding window before running any estimator.
        if bucket_state is not None:
            # The sanitizer resamples onto a fixed 100Hz grid and drops wall-clock
            # timestamps, so the window is trimmed by chunk count, not sample age.
            # The first tuple slot is reserved for timestamps if that changes.
            state_store.push_window_sample(
                bucket_state, None, payload['v_matrices'], payload['rssi'],
                window_chunks=WINDOW_CHUNKS
            )
            v_matrices = state_store.windowed_v_matrices(bucket_state)
            rssi_window = state_store.windowed_rssi(bucket_state)
        else:
            v_matrices = payload['v_matrices']
            rssi_window = payload['rssi']

        # Averaged in the linear (mW) domain: a mean of dB readings from a
        # fading signal sits below the dB of the true mean power (Jensen), which
        # reads as extra path loss and overestimates range downstream. rssi_std
        # feeds range_uncertainty_m in Stage 4, so a noisier chunk widens the
        # reported position uncertainty rather than being dropped.
        client_rssi = tx_power_estimator.mean_rssi_dbm(rssi_window)
        client_rssi_std = float(np.std(rssi_window))

        # Uses the carried-over VSS-LMS background when state is available,
        # a static per-chunk mean otherwise.
        ke = kinematic_tracker.kpvt_module(v_matrices, bucket_state)
        packets = v_matrices.shape[0]

        # Threshold is computed from the reference history BEFORE this chunk's
        # value is folded in, so the decision is not influenced by itself.
        # is_occupied_cfar stays None until enough history accumulates; Stage 4
        # falls back to the static KE_THRESHOLD until then.
        cfar_threshold, is_occupied_cfar = None, None
        if bucket_state is not None:
            cfar_threshold = kinematic_tracker.os_cfar_threshold(
                bucket_state["cfar_reference"], p_fa=CFAR_PFA
            )
            if cfar_threshold is not None:
                is_occupied_cfar = bool(ke > cfar_threshold)
            state_store.push_cfar_reference(bucket_state, ke, CFAR_REFERENCE_CELLS)

        ap_aod = None
        aod_confidence = None

        if nt < 2:
            routing, algo_name = "KPVT_ONLY", "NONE"
        else:
            routing = "KPVT_AND_SSE"
            if packets < STARVED_LIMIT: algo_name = "IAA_APES"
            elif ke < KE_THRESHOLD: algo_name = "SPOTFI"
            else: algo_name = "RES_2D_MUSIC" if nt >= 3 else "CA_ESPRIT"

            # confidence is HIGH when the eigenvalue spectrum supports the
            # rank-1 assumption the estimator relies on, LOW when a second
            # comparable path makes the angle a likely multipath blend.
            ap_aod, aod_confidence = spatial_algos.SSE_REGISTRY[algo_name](v_matrices, nt)

        results[bucket_key] = {
            "client_mac": client_mac,
            "ap_mac": ap_mac,
            "mimo": pkt_config,
            "routing": routing,
            "algorithm": algo_name,
            "kinematic_energy": ke,
            "cfar_threshold": cfar_threshold,
            "is_occupied_cfar": is_occupied_cfar,
            "ap_aod": ap_aod,
            "aod_confidence": aod_confidence,
            "client_rssi": client_rssi,
            "client_rssi_std": client_rssi_std
        }

    if state_file:
        state_store.save_state(state_file, state)

    with open(out_json, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    state_arg = sys.argv[3] if len(sys.argv) > 3 else None
    dispatcher(sys.argv[1], sys.argv[2], state_arg)