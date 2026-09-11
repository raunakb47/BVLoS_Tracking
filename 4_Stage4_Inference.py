#!/usr/bin/env python3
"""
Module: 4_Stage4_Inference.py
Localization representation placing the Monitor Card at (0,0) and the AP
on the positive Y-axis.
"""
import sys
import json
import os
import time
import numpy as np
from importlib import import_module

state_store = import_module('state_store')

KE_THRESHOLD = float(os.getenv("KE_THRESHOLD", 0.02))
WIFI_CHANNEL = int(os.getenv("WIFI_CHANNEL", 36))
TRACK_COAST_LIMIT = int(os.getenv("TRACK_COAST_LIMIT", state_store.DEFAULT_COAST_LIMIT))

# Monitor Card is ALWAYS the center of the grid
MC_COORDS = np.array([0.0, 0.0])

def channel_to_frequency(channel):
    """
    Converts IEEE 802.11 channel numbers to center frequency (MHz).
    Supports standard 2.4 GHz and 5 GHz bands.
    """
    if 1 <= channel <= 13:
        return 2407.0 + (5.0 * channel)
    elif channel == 14:
        return 2484.0
    elif 32 <= channel <= 177:
        return 5000.0 + (5.0 * channel)
    else:
        # Fallback to standard Ch 36 if an unsupported channel is provided
        return 5180.0 

def estimate_ap_baseline(ap_mac):
    """
    In real-time mode, establishes the Y-Axis baseline dynamically.
    """
    is_mobile = True if "HOT" in ap_mac or "MOB" in ap_mac else False
    return {"pos": np.array([0.0, 5.0]), "mobile": is_mobile}

def ray_circle_intersection(ap_pos, mc_pos, ap_aod, client_rssi, freq_mhz):
    """
    Geometrically intersects the AP's Angle Ray with the Monitor Card's RSSI Distance Circle.
    """
    # FSPL formula calculating radius using the dynamically generated frequency
    r = 10 ** ((27.55 - (20 * np.log10(freq_mhz)) + abs(client_rssi)) / 20.0)
    
    rad_ap = np.radians(ap_aod)
    D = np.array([np.sin(rad_ap), np.cos(rad_ap)])
    O = ap_pos - mc_pos
    
    b = 2.0 * np.dot(O, D)
    c = np.dot(O, O) - r**2
    delta = b**2 - 4*c
    
    if delta >= 0:
        t = (-b + np.sqrt(delta)) / 2.0
        t = max(t, 0.0)
    else:
        # If noise margin creates non-intersection, snap to the closest point on the ray
        t = max(-b / 2.0, 0.0)
        
    return ap_pos + (t * D), r

def calculate_dynamic_cep(ke, is_mobile):
    """ Dynamically models error radius based on physics and routing """
    base_error = 0.5
    mobility_penalty = 1.5 if is_mobile else 0.0
    kinematic_penalty = min(ke * 10, 2.0)
    return round(base_error + mobility_penalty + kinematic_penalty, 2)

def stage4_inference(stage3_json, state_file=None):
    # Perform channel to frequency translation once upon execution
    operating_freq = channel_to_frequency(WIFI_CHANNEL)

    with open(stage3_json, 'r') as f:
        results = json.load(f)

    state = state_store.load_state(state_file) if state_file else {}
    dashboard_state = {"Occupancy": 0, "Entities": []}
    seen_bucket_keys = []

    for bucket, data in results.items():
        ke = data["kinematic_energy"]
        # OS-CFAR (3_2_Kinematic_Tracker.py) gives each bucket its own dynamic
        # detection threshold once it has accumulated enough kinematic-energy
        # history; is_occupied_cfar is None before that history exists, in
        # which case the static global KE_THRESHOLD is used as the cold-start
        # fallback rather than leaving the bucket undetectable until then.
        is_occupied_cfar = data.get("is_occupied_cfar")
        is_occupied = is_occupied_cfar if is_occupied_cfar is not None else (ke > KE_THRESHOLD)
        if is_occupied: dashboard_state["Occupancy"] += 1

        entity = {
            "Mac": data["client_mac"],
            "State": "MOVING" if is_occupied else "STATIC",
            "Kinetic_Energy": round(ke, 4),
            "Detector": "OS-CFAR" if is_occupied_cfar is not None else "STATIC_THRESHOLD",
            "UI_Render": {}
        }

        # A LOW-confidence angle (3_1_Spatial_Algorithms._signal_subspace_confidence
        # detected a second comparably-strong path, so rank-1 does not hold this
        # chunk) is deliberately NOT turned into a position update: emitting one
        # anyway would silently blend two propagation paths into a single
        # confident-looking point. The bucket is left out of seen_bucket_keys so
        # the coast/hold logic below treats this chunk exactly like a chunk with
        # no packets at all -- KPVT's occupancy read is still reported, SSE just
        # sits this one out.
        if data["routing"] == "KPVT_AND_SSE" and data.get("aod_confidence") != "LOW":
            seen_bucket_keys.append(bucket)
            ap_mac = data["ap_mac"]
            ap_info = estimate_ap_baseline(ap_mac)
            ap_pos = ap_info["pos"]

            # Pass the derived frequency into the intersection engine
            client_coords, mc_distance = ray_circle_intersection(
                ap_pos, MC_COORDS, data["ap_aod"], data["client_rssi"], operating_freq
            )

            cep_radius = calculate_dynamic_cep(ke, ap_info["mobile"])

            entity["UI_Render"] = {
                "Tracking_Type": "Ray_Circle_Intersection",
                "Algorithm": data["algorithm"],
                "Anchor_MAC": ap_mac,
                "Anchor_Coords": [round(ap_pos[0], 2), round(ap_pos[1], 2)],
                "Client_Coords": [round(client_coords[0], 2), round(client_coords[1], 2)],
                "Uncertainty_Radius": cep_radius,
                "Vectors": {
                    "AP_AoD": round(data["ap_aod"], 1),
                    "MC_Distance_m": round(mc_distance, 2)
                },
                "Track_Status": "CONFIRMED"
            }

            # Remember this fix so a subsequent chunk with zero packets for this
            # bucket can still render a (aging) marker instead of the client
            # abruptly disappearing from the map.
            if state_file:
                state_store.mark_fix(state_store.get_bucket(state, bucket), client_coords, data["ap_aod"])

        dashboard_state["Entities"].append(entity)

    # BFI arrives on whatever cadence the AP happens to sound its clients at;
    # a bucket producing zero packets for a chunk or two is normal, not a
    # dropout. Rather than let such a client vanish from the map (or crash on
    # a KeyError further down the pipeline), hold it at its last confirmed
    # position -- the standard radar "coast" pattern: predict/hold across
    # missed detections, drop the track only after TRACK_COAST_LIMIT
    # consecutive misses. The UI is expected to render a COASTING entity
    # visibly stale (e.g. greyed out) using Age_Seconds, and to snap it back
    # to a solid marker the moment a fresh CONFIRMED fix reappears.
    if state_file:
        surviving = state_store.touch_buckets(state, seen_bucket_keys, coast_limit=TRACK_COAST_LIMIT)
        for bucket_key in surviving:
            if bucket_key in seen_bucket_keys:
                continue
            bucket_state = state[bucket_key]
            if bucket_state["track_status"] != "COASTING":
                continue
            client_mac, ap_mac, pkt_config = bucket_key.split('_')
            age_s = time.time() - bucket_state["last_seen_ts"]
            dashboard_state["Entities"].append({
                "Mac": client_mac,
                "State": "STALE",
                "Kinetic_Energy": None,
                "UI_Render": {
                    "Tracking_Type": "Ray_Circle_Intersection",
                    "Anchor_MAC": ap_mac,
                    "Client_Coords": bucket_state["last_position"],
                    "Vectors": {"AP_AoD": bucket_state["last_ap_aod"]},
                    "Track_Status": "COASTING",
                    "Age_Seconds": round(age_s, 1)
                }
            })
        state_store.save_state(state_file, state)

    print(json.dumps(dashboard_state, indent=2))

if __name__ == "__main__":
    state_arg = sys.argv[2] if len(sys.argv) > 2 else None
    stage4_inference(sys.argv[1], state_arg)