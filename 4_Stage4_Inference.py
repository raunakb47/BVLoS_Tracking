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
hotspot_classifier = import_module('hotspot_classifier')
tx_power_estimator = import_module('tx_power_estimator')
ap_registry = import_module('ap_registry')

KE_THRESHOLD = float(os.getenv("KE_THRESHOLD", 0.02))
WIFI_CHANNEL = int(os.getenv("WIFI_CHANNEL", 36))
TRACK_COAST_LIMIT = int(os.getenv("TRACK_COAST_LIMIT", state_store.DEFAULT_COAST_LIMIT))

# Log-distance path loss exponent (tx_power_estimator.log_distance_m). 2.0 is
# free space (FSPL, this framework's original/blind default); real indoor
# NLOS propagation runs higher (see log_distance_m's docstring) and a fixed
# n=2 assumption systematically overestimates range through walls -- set
# this once real per-site calibration is available.
PATH_LOSS_EXPONENT = float(os.getenv("PATH_LOSS_EXPONENT", tx_power_estimator.DEFAULT_PATH_LOSS_EXPONENT))

# Optional path to a real (oui_hex, vendor_name) CSV export -- see
# hotspot_classifier.load_mobile_oui_table(). Unset by default: the
# classifier's built-in table is illustrative only (a handful of verified
# entries out of the 1500+/900+ blocks Apple/Samsung alone hold), so mobile-
# hotspot classification is UNKNOWN for most real devices until a real table
# is configured here.
MOBILE_OUI_TABLE_PATH = os.getenv("MOBILE_OUI_TABLE_PATH")
_MOBILE_OUI_TABLE = (
    hotspot_classifier.load_mobile_oui_table(MOBILE_OUI_TABLE_PATH)
    if MOBILE_OUI_TABLE_PATH else None
)

# Path to the persistent, cross-chunk AP metadata registry (SSID, transmit
# power, beacon-RSSI-derived distance) that 2_Stage2_Extraction.sh maintains
# via extract_ap_metadata.py + ap_registry.py once Beacon/Probe Response
# capture is enabled (#5/#6). Read fresh each invocation since this process
# does not own writing it -- Stage 2's watchdog does, once per chunk.
AP_REGISTRY_PATH = os.getenv("AP_REGISTRY_PATH")

# Fallback AP-Monitor Card distance (meters) used only until a real
# measurement is available (see estimate_ap_baseline): this is a placeholder,
# not a calibrated value, and should not be trusted as ground truth.
DEFAULT_AP_DISTANCE_M = float(os.getenv("DEFAULT_AP_DISTANCE_M", 5.0))

# Monitor Card is ALWAYS the center of the grid
MC_COORDS = np.array([0.0, 0.0])

def estimate_ap_baseline(ap_mac, ssid=None, ap_distance_m=None):
    """
    In real-time mode, establishes the Y-Axis baseline dynamically.

    The coordinate system is defined with the AP on the positive Y-axis by
    convention (a purely local/ego-centric frame anchored on the monitor
    card, not an absolute compass bearing), so only the AP's *distance* from
    the monitor card needs to be established -- no angle-of-arrival at the
    monitor card is required. ap_distance_m is accepted for forward
    compatibility with an AP-beacon-RSSI-derived range measurement (an
    independent third leg of the localization triangle -- Client-AP bearing
    from BFI, Client-MonitorCard range from client_rssi, and this
    AP-MonitorCard range from the AP's own overheard Beacon RSSI -- see the
    #5/#6 capture-filter work); until that capture change lands and supplies
    a real value, this falls back to DEFAULT_AP_DISTANCE_M, an admitted
    placeholder rather than a measurement.

    "mobile" is a best-effort classification (see hotspot_classifier.py),
    not a confirmed device type: the previous check here ("HOT"/"MOB"
    substrings of a hex MAC address) could never match anything, so every
    link was silently treated as a fixed AP regardless of what it actually
    was. ssid is likewise accepted for forward compatibility with
    Beacon/Probe Response capture and is None until that lands.
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
    Geometrically intersects the AP's Angle Ray with the Monitor Card's RSSI Distance Circle.
    """
    # tx_power_dbm comes from tx_power_estimator.py rather than being assumed
    # to be 0 dBm, which the previous abs(client_rssi)-as-path-loss form
    # implicitly did. PATH_LOSS_EXPONENT lets a calibrated deployment correct
    # for real indoor/NLOS loss instead of the free-space default.
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
    Dynamically models error radius based on physics and routing.

    range_uncertainty_m (tx_power_estimator.range_uncertainty_m, first-order
    propagation of this chunk's RSSI standard deviation through the ranging
    formula) is a new term: previously the uncertainty radius only reflected
    motion/mobility, so a bucket with a rock-solid RSSI reading and one with
    RSSI swinging over several dB from fading reported the identical
    confidence circle. A noisier ranging input now widens the reported
    circle instead of that being silently absorbed into the point estimate.
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
            ap_record = registry.get(ap_mac, {})
            ap_info = estimate_ap_baseline(
                ap_mac, ssid=ap_record.get("ssid"), ap_distance_m=ap_record.get("ap_distance_m")
            )
            ap_pos = ap_info["pos"]

            # client_rssi is the RSSI of the CLIENT's own Compressed Beamforming
            # Report frame (sent client -> AP, overheard at the monitor card), so
            # ranging on it needs the client's typical transmit power, not the AP's.
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
                # Surfaces which parts of this geometry are actual measurements
                # vs. literature-typical/placeholder fallbacks, rather than
                # presenting a single point with no indication of provenance.
                "Calibration": {
                    "Client_TX_Power_dBm": client_tx_power_dbm,
                    "Client_TX_Power_Source": tx_power_source,
                    "AP_Distance_Is_Measured": ap_info["distance_is_measured"],
                    "Path_Loss_Exponent": PATH_LOSS_EXPONENT,
                    "Range_Uncertainty_m": round(range_uncertainty, 2),
                },
                # is_mobile is a best-effort guess (hotspot_classifier.py), surfaced
                # with its own confidence/reason rather than presented as fact -- a
                # dashboard should visibly distinguish this from a measured quantity.
                "Anchor_Mobility": {
                    "is_mobile": ap_info["mobile"],
                    "confidence": ap_info["mobility_confidence"],
                    "reason": ap_info["mobility_reason"],
                }
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