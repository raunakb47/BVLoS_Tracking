#!/usr/bin/env python3
"""
Module: 4_Stage4_Inference.py
Turn Stage 3's per-bucket angles and RSSI into dashboard coordinates, in an
ego-centric frame with the Monitor Card at (0,0) and the AP on the positive
Y-axis. Also owns track continuity (TRACK_STATE_FILE), which must be a
different file from Stage 3's STATE_FILE -- see state_store.py.
"""
import sys
import json
import os
import time
import numpy as np
from importlib import import_module

state_store = import_module('state_store')
hotspot_classifier = import_module('hotspot_classifier')
tx_power_estimator = import_module('tx_power_estimator')
ap_registry = import_module('ap_registry')

KE_THRESHOLD = float(os.getenv("KE_THRESHOLD", 0.02))
WIFI_CHANNEL = int(os.getenv("WIFI_CHANNEL", 36))
TRACK_COAST_LIMIT = int(os.getenv("TRACK_COAST_LIMIT", state_store.DEFAULT_COAST_LIMIT))

# Log-distance path loss exponent. 2.0 is free space; indoor NLOS runs higher,
# and n=2 through walls systematically overestimates range. Set per site once
# calibration data exists.
PATH_LOSS_EXPONENT = float(os.getenv("PATH_LOSS_EXPONENT", tx_power_estimator.DEFAULT_PATH_LOSS_EXPONENT))

# Path to a real (oui_hex, vendor_name) CSV export. Unset by default: the
# built-in table holds a handful of verified entries against the 1500+/900+
# blocks Apple and Samsung alone register, so classification is UNKNOWN for
# most devices until a full table is configured here.
MOBILE_OUI_TABLE_PATH = os.getenv("MOBILE_OUI_TABLE_PATH")
_MOBILE_OUI_TABLE = (
    hotspot_classifier.load_mobile_oui_table(MOBILE_OUI_TABLE_PATH)
    if MOBILE_OUI_TABLE_PATH else None
)

# Cross-chunk AP registry (SSID, transmit power, beacon-derived distance)
# written by Stage 2 once per chunk. Read fresh each invocation; this process
# does not write it.
AP_REGISTRY_PATH = os.getenv("AP_REGISTRY_PATH")

# Fallback AP-to-Monitor-Card distance (m), used until a beacon-derived
# measurement exists. A placeholder, not a calibrated value.
DEFAULT_AP_DISTANCE_M = float(os.getenv("DEFAULT_AP_DISTANCE_M", 5.0))

# Monitor Card is ALWAYS the center of the grid
MC_COORDS = np.array([0.0, 0.0])

def estimate_ap_baseline(ap_mac, ssid=None, ap_distance_m=None):
    """
    Place the AP on the Y-axis baseline.

    The frame is ego-centric on the monitor card, not a compass bearing, so
    only the AP's distance is needed -- no angle-of-arrival at the monitor
    card. ap_distance_m is the beacon-RSSI-derived range from the registry and
    forms the third leg of the triangle (client-AP bearing from BFI,
    client-monitor range from client_rssi, AP-monitor range from beacons);
    falls back to the DEFAULT_AP_DISTANCE_M placeholder when absent.

    "mobile" is a best-effort classification (hotspot_classifier.py), not a
    confirmed device type.
    """
    is_mobile, mobility_confidence, mobility_reason = hotspot_classifier.classify_beamformer(
        ap_mac, ssid=ssid, oui_table=_MOBILE_OUI_TABLE
    )
    distance = ap_distance_m if ap_distance_m is not None else DEFAULT_AP_DISTANCE_M
    return {
        "pos": np.array([0.0, distance]),
        "distance_is_measured": ap_distance_m is not None,
        "mobile": is_mobile,
        "mobility_confidence": mobility_confidence,
        "mobility_reason": mobility_reason,
    }

def ray_circle_intersection(ap_pos, mc_pos, ap_aod, client_rssi, freq_mhz, tx_power_dbm):
    """
    Intersect the AP's angle ray with the monitor card's RSSI distance circle.
    Returns (client_position, monitor_card_range_m).
    """
    r = tx_power_estimator.log_distance_m(client_rssi, tx_power_dbm, freq_mhz, PATH_LOSS_EXPONENT)

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

def calculate_dynamic_cep(ke, is_mobile, range_uncertainty_m=0.0):
    """
    Error radius (m) from motion, anchor mobility and ranging noise.

    range_uncertainty_m propagates this chunk's RSSI standard deviation through
    the ranging formula to first order, so a bucket whose RSSI is swinging
    several dB reports a wider circle than one with a steady reading.
    """
    base_error = 0.5
    mobility_penalty = 1.5 if is_mobile else 0.0
    kinematic_penalty = min(ke * 10, 2.0)
    return round(base_error + mobility_penalty + kinematic_penalty + range_uncertainty_m, 2)

def stage4_inference(stage3_json, state_file=None):
    # Perform channel to frequency translation once upon execution
    operating_freq = tx_power_estimator.channel_to_frequency(WIFI_CHANNEL)

    with open(stage3_json, 'r') as f:
        results = json.load(f)

    state = state_store.load_state(state_file) if state_file else {}
    registry = ap_registry.load_registry(AP_REGISTRY_PATH) if AP_REGISTRY_PATH else {}
    dashboard_state = {"Occupancy": 0, "Entities": []}
    seen_bucket_keys = []
    entity_by_bucket = {}

    for bucket, data in results.items():
        ke = data["kinematic_energy"]
        # OS-CFAR gives each bucket its own threshold once it has enough
        # history; None before then, so KE_THRESHOLD serves as the cold start.
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

        # A LOW-confidence angle is not turned into a position update: rank-1
        # does not hold this chunk, so the angle would blend two paths into one
        # confident-looking point. Excluded from seen_bucket_keys so the coast
        # logic below treats it like a chunk with no packets; KPVT's occupancy
        # read still stands.
        if data["routing"] == "KPVT_AND_SSE" and data.get("aod_confidence") != "LOW":
            seen_bucket_keys.append(bucket)
            ap_mac = data["ap_mac"]
            ap_record = registry.get(ap_mac, {})
            ap_info = estimate_ap_baseline(
                ap_mac, ssid=ap_record.get("ssid"), ap_distance_m=ap_record.get("ap_distance_m")
            )
            ap_pos = ap_info["pos"]

            # client_rssi is measured on the client's own Compressed Beamforming
            # Report, so ranging it needs the client's transmit power, not the AP's.
            client_tx_power_dbm, tx_power_source = tx_power_estimator.estimate_client_tx_power_dbm()

            client_coords, mc_distance = ray_circle_intersection(
                ap_pos, MC_COORDS, data["ap_aod"], data["client_rssi"], operating_freq, client_tx_power_dbm
            )

            range_uncertainty = tx_power_estimator.range_uncertainty_m(
                mc_distance, data.get("client_rssi_std", 0.0), PATH_LOSS_EXPONENT
            )
            cep_radius = calculate_dynamic_cep(ke, ap_info["mobile"], range_uncertainty)

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
                "Track_Status": "CONFIRMED",
                # Marks which parts of this geometry are measured and which are
                # literature-typical defaults, so the point is not presented
                # without provenance.
                "Calibration": {
                    "Client_TX_Power_dBm": client_tx_power_dbm,
                    "Client_TX_Power_Source": tx_power_source,
                    "AP_Distance_Is_Measured": ap_info["distance_is_measured"],
                    "Path_Loss_Exponent": PATH_LOSS_EXPONENT,
                    "Range_Uncertainty_m": round(range_uncertainty, 2),
                },
                # Best-effort guess with its own confidence and reason attached;
                # a dashboard should render it differently from a measurement.
                "Anchor_Mobility": {
                    "is_mobile": ap_info["mobile"],
                    "confidence": ap_info["mobility_confidence"],
                    "reason": ap_info["mobility_reason"],
                }
            }

            # Remembered so a later chunk with no packets for this bucket can
            # still render an aging marker instead of the client vanishing.
            if state_file:
                state_store.mark_fix(state_store.get_bucket(state, bucket), client_coords, data["ap_aod"])

        # Indexed by bucket so the coast pass can fill in a held position for a
        # bucket present in this chunk but without a usable angle, rather than
        # appending a second entity for the same MAC.
        entity_by_bucket[bucket] = entity
        dashboard_state["Entities"].append(entity)

    # A bucket producing no packets for a chunk or two is normal on BFI
    # cadence, not a dropout, so hold it at its last confirmed position and
    # drop only after TRACK_COAST_LIMIT consecutive misses. The UI is expected
    # to grey a COASTING entity by Age_Seconds and snap it back on the next
    # CONFIRMED fix.
    if state_file:
        surviving = state_store.touch_buckets(state, seen_bucket_keys, coast_limit=TRACK_COAST_LIMIT)
        for bucket_key in surviving:
            bucket_state = state[bucket_key]
            if bucket_state["track_status"] != "COASTING":
                continue
            client_mac, ap_mac, pkt_config = bucket_key.split('_')
            coast_render = {
                "Tracking_Type": "Ray_Circle_Intersection",
                "Anchor_MAC": ap_mac,
                "Client_Coords": bucket_state["last_position"],
                "Vectors": {"AP_AoD": bucket_state["last_ap_aod"]},
                "Track_Status": "COASTING",
                "Age_Seconds": round(time.time() - bucket_state["last_seen_ts"], 1)
            }

            existing = entity_by_bucket.get(bucket_key)
            if existing is not None:
                # Packets this chunk but no accepted angle (LOW SSE confidence).
                # Keep the entity's KPVT-derived State and attach the held
                # position; an empty UI_Render would blink the client off the
                # map on exactly the chunks KPVT is meant to carry.
                existing["UI_Render"] = coast_render
                continue

            dashboard_state["Entities"].append({
                "Mac": client_mac,
                "State": "STALE",
                "Kinetic_Energy": None,
                "UI_Render": coast_render
            })
        state_store.save_state(state_file, state)

    print(json.dumps(dashboard_state, indent=2))

if __name__ == "__main__":
    state_arg = sys.argv[2] if len(sys.argv) > 2 else None
    stage4_inference(sys.argv[1], state_arg)