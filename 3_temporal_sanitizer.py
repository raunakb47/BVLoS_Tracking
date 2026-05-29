import numpy as np
import sys
from scipy import interpolate, signal

def sanitize_bfi_data(raw_npy_file):
    try:
        raw_data = np.load(raw_npy_file, allow_pickle=True).item()
    except Exception as e:
        return

    sanitized_data = {}

    for target_key, temporal_list in raw_data.items():
        if len(temporal_list) < 30: continue

        timestamps = np.array([item[0] for item in temporal_list])
        v_matrices = np.array([item[1] for item in temporal_list])
        
        t_relative = timestamps - timestamps[0]
        duration = t_relative[-1]
        if duration == 0: continue
        
        # 100 Hz Uniform Analytical Grid
        uniform_time_grid = np.linspace(0, duration, max(int(duration * 100), 10))
        
        original_shape = v_matrices.shape
        v_flat = v_matrices.reshape(len(timestamps), -1)
        
        # Jitter Correction
        interpolator = interpolate.interp1d(t_relative, v_flat, axis=0, kind='cubic', fill_value="extrapolate")
        v_uniform_flat = interpolator(uniform_time_grid)
        
        # Quantization Smoothing
        window = min(15, len(uniform_time_grid))
        if window % 2 == 0: window -= 1
        poly = min(3, window - 1)
        
        v_smoothed_flat = signal.savgol_filter(v_uniform_flat, window_length=window, polyorder=poly, axis=0)
        v_sanitized = v_smoothed_flat.reshape(len(uniform_time_grid), *original_shape[1:])
        
        sanitized_data[target_key] = v_sanitized

    output_file = raw_npy_file.replace('_raw.npy', '_sanitized.npy')
    np.save(output_file, sanitized_data)

if __name__ == "__main__":
    sanitize_bfi_data(sys.argv[1])