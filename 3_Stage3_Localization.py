#!/usr/bin/env python3
"""
Module: 3_Stage3_Localization.py
Evaluates matrix properties and routes them to the appropriate algorithm.
"""
import sys
import os
import json
import numpy as np
from sklearn.decomposition import PCA
from importlib import import_module

spatial_algos = import_module('3_1_Spatial_Algorithms')

KE_THRESHOLD = float(os.getenv("KE_THRESHOLD", 0.02))
STARVED_LIMIT = int(os.getenv("PACKET_STARVATION_LIMIT", 15))

def kpvt_module(v_matrices):
    t_steps = v_matrices.shape[0]
    v_abs = np.abs(v_matrices).reshape(t_steps, -1)
    residual = v_abs - np.mean(v_abs, axis=0)
    variance_profile = PCA(n_components=1).fit_transform(residual).flatten()
    return float(np.var(variance_profile))

def dispatcher(sanitized_file, out_json):
    data = np.load(sanitized_file, allow_pickle=True).item()
    results = {}

    for bucket_key, payload in data.items():
        client_mac, ap_mac, pkt_config = bucket_key.split('_')
        nt = int(pkt_config.split('x')[0]) 
        
        v_matrices = payload['v_matrices']
        client_rssi = float(np.mean(payload['rssi']))
        
        ke = kpvt_module(v_matrices)
        packets = v_matrices.shape[0]
        
        ap_aod = None
        
        if nt < 2: 
            routing, algo_name = "KPVT_ONLY", "NONE"
        else:
            routing = "KPVT_AND_SSE"
            if packets < STARVED_LIMIT: algo_name = "IAA_APES"
            elif ke < KE_THRESHOLD: algo_name = "SPOTFI"
            else: algo_name = "RES_2D_MUSIC" if nt >= 3 else "CA_ESPRIT"
                
            ap_aod = spatial_algos.SSE_REGISTRY[algo_name](v_matrices, nt)
            
        results[bucket_key] = {
            "client_mac": client_mac,
            "ap_mac": ap_mac,
            "mimo": pkt_config,
            "routing": routing,
            "algorithm": algo_name,
            "kinematic_energy": ke,
            "ap_aod": ap_aod,
            "client_rssi": client_rssi
        }

    with open(out_json, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    dispatcher(sys.argv[1], sys.argv[2])