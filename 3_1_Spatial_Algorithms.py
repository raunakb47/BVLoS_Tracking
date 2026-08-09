#!/usr/bin/env python3
"""
Module: 3_1_Spatial_Algorithms.py
Spatial Subspace Extractor (SSE) algorithms.
"""
import numpy as np
from scipy import linalg

def algo_ca_esprit(v_matrices, nt):
    """ CA-ESPRIT Analytical solver. Conjugate augmented for low-aperture. """
    v_avg = np.mean(v_matrices, axis=0) 
    z = v_avg[:, :, 0].T  
    
    if nt == 2:
        z_conj = np.dot(np.array([[0, 1], [1, 0]]), np.conj(z))
        y = np.vstack([z, z_conj]) 
    else: y = z

    R = np.dot(y, np.conj(y).T) / v_matrices.shape[1]
    vals, vecs = linalg.eigh(R)
    u_s = vecs[:, -1:]
    
    phi = np.dot(linalg.pinv(u_s[:-1, :]), u_s[1:, :])
    ap_aod = np.degrees(np.arcsin(np.clip(np.angle(linalg.eigvals(phi)[0]) / np.pi, -1, 1)))
    return float(ap_aod)

def algo_spotfi(v_matrices, nt):
    """ Spot-Fi Forward-Backward smoothing. Decorrelates indoor coherent multipath. """
    v_avg = np.mean(v_matrices, axis=0)
    l_w = max(10, v_avg.shape[0] // 2)
    
    sub_matrices = [v_avg[i:i+l_w, :, 0].flatten() for i in range(v_avg.shape[0] - l_w + 1)]
    R_smooth = np.dot(np.array(sub_matrices).T.conj(), np.array(sub_matrices)) / len(sub_matrices)
    
    vals, vecs = linalg.eigh(R_smooth)
    noise_sub = vecs[:, :-1]
    
    grid = np.linspace(-np.pi/2, np.pi/2, 181)
    spectrum = [1.0 / np.real(np.dot(np.dot(noise_sub.T.conj(), np.exp(-1j * np.pi * np.arange(R_smooth.shape[0]) * np.sin(th))).T.conj(), np.dot(noise_sub.T.conj(), np.exp(-1j * np.pi * np.arange(R_smooth.shape[0]) * np.sin(th))))) for th in grid]
    
    return float(np.degrees(grid[np.argmax(spectrum)]))

def algo_iaa_apes(v_matrices, nt):
    """ IAA-APES Iterative solver. Single-snapshot execution for mobile hotspots. """
    v_snap = np.mean(v_matrices[-1], axis=0)[:, 0]
    m = len(v_snap)
    grid = np.linspace(-np.pi/2, np.pi/2, 91)
    p_spec = np.ones(len(grid))
    
    for _ in range(3): 
        R = sum([p_spec[i] * np.dot(np.exp(-1j * np.pi * np.arange(m) * np.sin(th)).reshape(-1, 1), np.exp(-1j * np.pi * np.arange(m) * np.sin(th)).reshape(-1, 1).conj().T) for i, th in enumerate(grid)])
        R_inv = linalg.pinv(R + 1e-3 * np.eye(m))
        for i, th in enumerate(grid):
            a = np.exp(-1j * np.pi * np.arange(m) * np.sin(th)).reshape(-1, 1)
            denom = float(np.real(np.dot(np.dot(a.conj().T, R_inv), a)[0, 0]))
            if denom > 0: p_spec[i] = np.abs(np.dot(np.dot(a.conj().T, R_inv), v_snap.reshape(-1, 1))[0, 0] / denom)**2

    return float(np.degrees(grid[np.argmax(p_spec)]))

def algo_res_2d_music(v_matrices, nt):
    """ Residual 2D MUSIC for clean arrays >= 3x3 """
    v_avg = np.mean(v_matrices, axis=0) 
    z = v_avg[:, :, 0].T  
    R = np.dot(z, np.conj(z).T) / v_matrices.shape[1]
    
    vals, vecs = linalg.eigh(R)
    noise_sub = vecs[:, :-1]
    
    grid = np.linspace(-np.pi/2, np.pi/2, 181)
    spectrum = [1.0 / np.real(np.dot(np.dot(noise_sub.T.conj(), np.exp(-1j * np.pi * np.arange(nt) * np.sin(th))).T.conj(), np.dot(noise_sub.T.conj(), np.exp(-1j * np.pi * np.arange(nt) * np.sin(th))))) for th in grid]
    
    return float(np.degrees(grid[np.argmax(spectrum)]))

SSE_REGISTRY = {
    "CA_ESPRIT": algo_ca_esprit,
    "SPOTFI": algo_spotfi,
    "IAA_APES": algo_iaa_apes,
    "RES_2D_MUSIC": algo_res_2d_music
}