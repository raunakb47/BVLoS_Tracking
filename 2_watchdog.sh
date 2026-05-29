#!/bin/bash
# /workspace/BVLoS_Live_Tracker/2_watchdog.sh

source ./config.env

WATCH_DIR="./live_chunks"
WIBFI_DIR="../Wi-BFI" 
LOG_FILE="./pipeline_errors.log"

echo "[*] Universal ISAC Pipeline Active. Monitoring spool directory..."
echo "[*] Extraction errors will be routed to $LOG_FILE"

inotifywait -m -e close_write --format "%w%f" $WATCH_DIR | while read NEW_PCAP
do
    BASE_NAME=$(basename "$NEW_PCAP" .pcap)
    
    RAW_VMATRIX="${WATCH_DIR}/${BASE_NAME}_vmatrix.npy"
    RAW_ANGLES="${WATCH_DIR}/${BASE_NAME}_angles.npy"
    SANITIZED_FILE="${WATCH_DIR}/${BASE_NAME}_sanitized.npy"
    
    # Stage 1: Protocol Extraction 
    # We capture standard error to a log instead of /dev/null
    python $WIBFI_DIR/main.py \
        "$NEW_PCAP" \
        "$WIFI_STANDARD" \
        "$MIMO_MODE" \
        "$FALLBACK_CONFIG" \
        "$BANDWIDTH" \
        "$MAX_PACKETS" \
        "$RAW_VMATRIX" \
        "$RAW_ANGLES" >> "$LOG_FILE" 2>&1
    
    # Verify the V-Matrix was successfully extracted before proceeding
    if [ -f "$RAW_VMATRIX" ]; then
        
        # Stage 2: Temporal Interpolation (CSMA/CA Jitter Correction)
        python 3_temporal_sanitizer.py "$RAW_VMATRIX" >> "$LOG_FILE" 2>&1
        
        if [ -f "$SANITIZED_FILE" ]; then
            # Terminal clear for continuous real-time readout
            printf '\033[2J\033[H'
            
            # Stage 3: Dual-Heuristic Analytics
            python 4_dual_heuristic_tracker.py "$SANITIZED_FILE"
            
            # Cleanup
            rm "$RAW_VMATRIX" "$RAW_ANGLES" "$SANITIZED_FILE"
        fi
    fi
    
    # Clean up the raw PCAP chunk after processing to prevent reprocessing and manage storage
    rm "$NEW_PCAP"
done