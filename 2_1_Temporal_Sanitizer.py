#!/usr/bin/env python3
"""
Module: 2_1_Temporal_Sanitizer.py
Enforce the Temporal Dropout Threshold (TDT) and uniformly maps tensors and RSSI.
"""
import sys
import numpy as np
from scipy import interpolate

def enforce_tdt_and_sanitize(raw_npy, tdt_ms):
    data = np.load(raw_npy, allow_pickle=True).item()
    sanitized = {}
    tdt_sec = float(tdt_ms) / 1000.0

    for bucket_key, packets in data.items():
        if len(packets) < 4: continue
            
        timestamps = np.array([p[0] for p in packets])
        v_matrices = np.array([p[1] for p in packets], dtype=complex)
        rssi_vals = np.array([p[2] for p in packets], dtype=float)
        
        time_diffs = np.diff(timestamps)
        valid_segments, current_seg = [], [0]
        
        for i, diff in enumerate(time_diffs):
            if diff > tdt_sec:
                valid_segments.append(current_seg)
                current_seg = [i + 1]
            else:
                current_seg.append(i + 1)
        valid_segments.append(current_seg)
        
        sanitized_v, sanitized_rssi = [], []
        
        for seg in valid_segments:
            if len(seg) < 4: continue 
            
            t_seg = timestamps[seg]
            v_seg = v_matrices[seg].reshape(len(seg), -1)
            r_seg = rssi_vals[seg]
            
            t_grid = np.linspace(t_seg[0], t_seg[-1], max(10, int((t_seg[-1] - t_seg[0]) * 100)))
            
            interp_r = interpolate.interp1d(t_seg, np.real(v_seg), axis=0, kind='cubic')
            interp_i = interpolate.interp1d(t_seg, np.imag(v_seg), axis=0, kind='cubic')
            interp_rssi = interpolate.interp1d(t_seg, r_seg, kind='linear')
            
            v_smooth = (interp_r(t_grid) + 1j * interp_i(t_grid)).reshape(len(t_grid), *v_matrices.shape[1:])
            
            sanitized_v.append(v_smooth)
            sanitized_rssi.append(interp_rssi(t_grid))
            
        if sanitized_v:
            sanitized[bucket_key] = {
                'v_matrices': np.concatenate(sanitized_v, axis=0),
                'rssi': np.concatenate(sanitized_rssi, axis=0)
            }

    np.save(raw_npy.replace('_vmatrix.npy', '_sanitized.npy'), sanitized)

if __name__ == "__main__":
    enforce_tdt_and_sanitize(sys.argv[1], sys.argv[2])