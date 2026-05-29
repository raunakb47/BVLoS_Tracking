import numpy as np
import sys
from sklearn.decomposition import PCA
from scipy import linalg

def vss_lms_background_subtraction(v_matrices, mu_max=0.1, alpha=0.01):
    """
    Adaptive filter tracking static room boundaries and AP ego-motion.
    Outputs the dynamic target residual (e(n)).
    """
    t_steps, subcarriers, nt, nr = v_matrices.shape
    v_flat = np.abs(v_matrices).reshape(t_steps, -1)
    
    background_w = np.zeros(v_flat.shape[1], dtype=float)
    dynamic_residual = np.zeros_like(v_flat)
    mu = mu_max
    
    for t in range(t_steps):
        signal = v_flat[t]
        error = signal - background_w
        dynamic_residual[t] = error
        
        # Weight update
        background_w += mu * error
        
        # Variable Step Size: Expands to track fast walking, shrinks to detect breathing
        error_power = np.mean(error**2)
        mu = np.clip(mu + alpha * error_power, 0.001, mu_max)
        
    return dynamic_residual.reshape(t_steps, subcarriers, nt, nr)

def calculate_esprit_aod(dynamic_tensor):
    """
    Calculates ESPRIT on the dynamic target subspace.
    Strictly requires Nt >= 3 to maintain rotational invariance.
    """
    packets, subcarriers, nt, nr = dynamic_tensor.shape
    v_spatial = np.mean(dynamic_tensor, axis=1) # Average across frequency
    
    R = np.zeros((nt, nt), dtype=complex)
    for i in range(packets):
        h_vec = v_spatial[i, :, 0].reshape(-1, 1)
        R += np.dot(h_vec, h_vec.conj().T)
    R /= packets
    
    vals, vecs = linalg.eigh(R)
    U_s = vecs[:, -1:] 
    
    U1 = U_s[:-1, :]
    U2 = U_s[1:, :]
    
    phi = np.dot(linalg.pinv(U1), U2)
    phase_shift = np.angle(linalg.eigvals(phi)[0])
    
    return np.degrees(np.arcsin(np.clip(phase_shift / np.pi, -1, 1)))

def dual_heuristic_engine(sanitized_file):
    print("==================================================")
    print("DUAL-HEURISTIC TRACKING")
    print("==================================================")
    
    data = np.load(sanitized_file, allow_pickle=True).item()
    
    for target_key, v_matrices in data.items():
        time_steps, subcarriers, nt, nr = v_matrices.shape
        
        # ----------------------------------------------------
        # HEURISTIC 1: UNIVERSAL KINEMATIC CORE (VSS-LMS)
        # ----------------------------------------------------
        # Strips static environment / Hotspot movement
        dynamic_tensor = vss_lms_background_subtraction(v_matrices)
        
        # Flatten the dynamic residual to calculate energy via PCA
        pca = PCA(n_components=1)
        residual_flat = dynamic_tensor.reshape(time_steps, -1)
        kinematic_variance = pca.fit_transform(residual_flat).flatten()
        energy = np.var(kinematic_variance)
        
        state = "OCCUPIED (Moving Target)" if energy > 0.02 else " STATIC (Environment Empty)"
        
        print(f" [*] Link Target   : {target_key.split('_')[0]}")
        print(f"     Geometry      : {nt}x{nr} MIMO")
        print(f"     Kinematic Node: {state} (KE: {energy:.4f})")
        
        # ----------------------------------------------------
        # HEURISTIC 2: OPPORTUNISTIC SPATIAL OVERLAY (ESPRIT)
        # ----------------------------------------------------
        # If the AP hardware is robust enough (e.g., Commercial Router)
        if nt >= 3:
            aoa = calculate_esprit_aod(dynamic_tensor)
            print(f"     Spatial Vector: {aoa:+.2f}° (Relative to AP)")
        else:
            # If the AP is a smartphone or lightweight hardware
            print(f"     Spatial Vector: [BLOCKED] Hardware physically incapable (Nt={nt})")
            
        print("--------------------------------------------------")

if __name__ == "__main__":
    dual_heuristic_engine(sys.argv[1])