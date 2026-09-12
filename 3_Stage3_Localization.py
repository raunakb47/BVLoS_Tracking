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

        # Fold this chunk's samples into the bucket's sliding window before
        # running any estimator, so KPVT/SSE see up to WINDOW_CHUNKS chunks
        # of history instead of only the chunk that just arrived.
        if bucket_state is not None:
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

        # Averaging RSSI in the linear (mW) domain, not the raw dBm values,
        # avoids a real bias: dB is a concave (log) transform of power, so a
        # naive arithmetic mean of several dB readings from a fading signal
        # is always below the dB of the true average power (Jensen's
        # inequality) -- read here as extra path loss, and turned into an
        # overestimated range downstream. rssi_std feeds range_uncertainty_m
        # in Stage 4 so a noisier (more-faded) chunk widens the reported
        # position uncertainty instead of that noise being silently dropped.
        client_rssi = tx_power_estimator.mean_rssi_dbm(rssi_window)
        client_rssi_std = float(np.std(rssi_window))

        # kpvt_module uses the bucket's carried-over VSS-LMS background estimate
        # when state is available (see 3_2_Kinematic_Tracker.py), falling back to
        # a static per-chunk mean otherwise.
        ke = kinematic_tracker.kpvt_module(v_matrices, bucket_state)
        packets = v_matrices.shape[0]

        # OS-CFAR: estimate this bucket's own dynamic detection threshold from
        # its recent kinematic-energy history *before* this chunk's value is
        # folded in, so the decision for this chunk isn't influenced by itself.
        # Runs alongside (not instead of) the static KE_THRESHOLD comparison
        # Stage 4 already makes: is_occupied_cfar is None (Stage 4 falls back
        # to the static threshold) until enough reference history accumulates,
        # then takes over as a per-bucket, self-calibrating alternative to one
        # constant applied uniformly across every bucket and environment.
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