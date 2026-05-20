import numpy as np
import sys

def clear_terminal():
    """Clears the terminal screen using high-speed ANSI escape codes."""
    print('\033[2J\033[H', end='', flush=True)

def calculate_1d_music(v_matrices):
    """
    Computes the 1D-MUSIC spatial pseudo-spectrum.
    
    Args:
        v_matrices (np.ndarray): Tensor of shape (Packets, Subcarriers, N_tx, N_rx)
                                 Note: Nr (rows) = N_tx, Nc (cols) = N_rx in IEEE 802.11 codebooks.
        
    Returns:
        tuple: (Search angles, Pseudo-spectrum power in dB, Transmit antenna count)
    """
    # Autonomously resolve the spatial dimensions from the tensor shape
    num_packets, num_subcarriers, n_tx, n_rx = v_matrices.shape
    
    # The mathematical resolution relies on the AP's transmitting elements (M)
    m_antennas = n_tx 

    # Step 1: Frequency-domain averaging to stabilize the temporal phase variations
    # Averages across the subcarrier dimension (axis=1)
    v_spatial = np.mean(v_matrices, axis=1) # New Shape: (Packets, N_tx, N_rx)
    
    # Isolate the primary spatial stream for line-of-sight tracking
    # We take all packets, all Tx antennas, but only the 0th spatial stream
    h_matrix = v_spatial[:, :, 0] # New Shape: (Packets, N_tx)
    
    # Step 2: Construct the Sample Spatial Covariance Matrix (R)
    R = np.zeros((m_antennas, m_antennas), dtype=complex)
    for i in range(num_packets):
        h_vec = h_matrix[i, :].reshape(-1, 1) 
        R += np.dot(h_vec, h_vec.conj().T)
    R = R / num_packets
    
    # Step 3: Eigenvalue Decomposition (EVD) for Subspace Partitioning
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    
    # Sort eigenvalues in descending order to identify the dominant subspace
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Isolate the Noise Subspace (E_n). Assumes a single dominant multipath trajectory (K=1) per device
    E_n = eigenvectors[:, 1:] 
    
    # Step 4: Spatial Sweep (Steering Vector Orthogonality)
    search_angles = np.arange(-90, 91, 1)
    pseudo_spectrum = np.zeros(len(search_angles))
    
    # Assumption: Standard Uniform Linear Array (ULA) with half-wavelength element spacing
    d_over_lambda = 0.5 
    
    for i, theta in enumerate(search_angles):
        rad = np.radians(theta)
        a_theta = np.array([
            np.exp(-1j * 2 * np.pi * d_over_lambda * m * np.sin(rad)) 
            for m in range(m_antennas)
        ]).reshape(-1, 1)
        
        # Calculate projection onto the noise subspace
        projection = np.dot(a_theta.conj().T, np.dot(E_n, np.dot(E_n.conj().T, a_theta)))
        
        # Format output logarithmically (dB)
        pseudo_spectrum[i] = 10 * np.log10(1.0 / (np.abs(projection[0, 0]) + 1e-10))
        
    return search_angles, pseudo_spectrum, m_antennas


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[!] Execution Error: Target matrix payload undefined.")
        sys.exit(1)
        
    matrix_file = sys.argv[1]
    
    try:
        # Load the dictionary of MAC-demultiplexed BFI matrices from Wi-BFI
        environmental_data = np.load(matrix_file, allow_pickle=True).item()
        
        # High-speed cross-platform terminal clear
        clear_terminal()
        
        print("==================================================")
        print(" ISAC PROMISCUOUS AoA BEARING RESOLVER")
        print("==================================================")
        print(f" Temporal Chunk : {matrix_file.split('/')[-1]}")
        print(f" Discovered     : {len(environmental_data)} radiating state(s)")
        print("--------------------------------------------------")
        
        # Iterate through every unique device/MIMO-state bucket
        for target_key, v_list in environmental_data.items():
            
            # Extract MAC and Antenna Config from the composite string (e.g., "AA:BB:CC_2x2")
            try:
                target_mac, array_topology = target_key.split('_')
            except ValueError:
                target_mac = target_key
                array_topology = "Unknown"

            # Discard non-stationary entities lacking mathematical correlation density
            if len(v_list) < 10:
                continue 
                
            # Convert the list of matrix arrays into a single 4D Numpy Tensor
            v_tensor = np.array(v_list)
            
            # Execute the localization engine natively based on tensor shape
            angles, spectrum, n_tx = calculate_1d_music(v_tensor)
            
            peak_index = np.argmax(spectrum)
            resolved_aoa = angles[peak_index]
            resolved_power = spectrum[peak_index]
            
            print(f" [*] Entity Target  : {target_mac}")
            print(f"     Angle of Arrival: {resolved_aoa}°")
            print(f"     Subspace Power  : {resolved_power:.2f} dB")
            print(f"     Array Topology  : {n_tx}-Element ULA ({array_topology} detected)")
            print("--------------------------------------------------")
            
        # Post-processing filesystem cleanup to prevent SSD overflow
        import os # Kept strictly for file deletion, not shell commands
        os.remove(matrix_file)
        
    except Exception as e:
        print(f"[!] Processing exception in {matrix_file}: {e}")