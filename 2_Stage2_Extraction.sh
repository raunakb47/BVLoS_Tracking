#!/bin/bash
# ==============================================================================
# Module: 2_Stage2_Extraction.sh
# Wi-BFI payload extraction and trigger subsequent stages
# ==============================================================================
source ./config.env

echo "[*] Stage 2: Extraction Dispatcher Active. Watching for chunks..."

inotifywait -m -e close_write --format "%w%f" "$WATCH_DIR" | while read -r NEW_PCAP
do
    if [[ "$NEW_PCAP" != *.pcap ]]; then continue; fi
    BASE=$(basename "$NEW_PCAP" .pcap)
    
    RAW_VMATRIX="${WATCH_DIR}/${BASE}_vmatrix.npy"
    RAW_ANGLES="${WATCH_DIR}/${BASE}_angles.npy"
    SANITIZED="${WATCH_DIR}/${BASE}_sanitized.npy"
    
    python3 "$WIBFI_DIR/main.py" "$NEW_PCAP" "$WIFI_STANDARD" "$MIMO_MODE" "$FALLBACK_CONFIG" "$BANDWIDTH" "$MAX_PACKETS" "$RAW_VMATRIX" "$RAW_ANGLES" >> "$LOG_FILE" 2>&1
    
    if [ -f "$RAW_VMATRIX" ]; then
        python3 2_1_Temporal_Sanitizer.py "$RAW_VMATRIX" "$TDT_MS" >> "$LOG_FILE" 2>&1
        
        if [ -f "$SANITIZED" ]; then
            # STATE_FILE carries per-bucket sliding-window and track-continuity
            # state across this chunk's Stage 3/4 subprocesses and the next.
            python3 3_Stage3_Localization.py "$SANITIZED" "$STAGE3_OUT" "$STATE_FILE"
            python3 4_Stage4_Inference.py "$STAGE3_OUT" "$STATE_FILE"
            rm -f "$RAW_VMATRIX" "$RAW_ANGLES" "$SANITIZED"
        fi
    fi
    rm -f "$NEW_PCAP"
done