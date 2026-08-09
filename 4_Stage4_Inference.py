#!/usr/bin/env python3
"""
Module: 4_Stage4_Inference.py
Localization representation placing the Monitor Card at (0,0) and the AP
on the positive Y-axis.
"""
import sys
import json
import os
import numpy as np

KE_THRESHOLD = float(os.getenv("KE_THRESHOLD", 0.02))
WIFI_CHANNEL = int(os.getenv("WIFI_CHANNEL", 36))

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

def stage4_inference(stage3_json):
    # Perform channel to frequency translation once upon execution
    operating_freq = channel_to_frequency(WIFI_CHANNEL)

    with open(stage3_json, 'r') as f:
        results = json.load(f)
        
    dashboard_state = {"Occupancy": 0, "Entities": []}
    
    for bucket, data in results.items():
        ke = data["kinematic_energy"]
        is_occupied = ke > KE_THRESHOLD
        if is_occupied: dashboard_state["Occupancy"] += 1
        
        entity = {
            "Mac": data["client_mac"],
            "State": "MOVING" if is_occupied else "STATIC",
            "Kinetic_Energy": round(ke, 4),
            "UI_Render": {}
        }
        
        if data["routing"] == "KPVT_AND_SSE":
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
                }
            }
                
        dashboard_state["Entities"].append(entity)

    print(json.dumps(dashboard_state, indent=2))

if __name__ == "__main__":
    stage4_inference(sys.argv[1])