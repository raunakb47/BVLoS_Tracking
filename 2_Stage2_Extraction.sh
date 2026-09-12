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
    AP_CHUNK_META="${WATCH_DIR}/${BASE}_ap_metadata.json"

    python3 "$WIBFI_DIR/main.py" "$NEW_PCAP" "$WIFI_STANDARD" "$MIMO_MODE" "$FALLBACK_CONFIG" "$BANDWIDTH" "$MAX_PACKETS" "$RAW_VMATRIX" "$RAW_ANGLES" >> "$LOG_FILE" 2>&1

    # Reads the same chunk for Beacon frames (now captured alongside BFI
    # action frames -- see CAPTURE_FILTER in config.env) and folds any
    # SSID/transmit-power/RSSI observed for each AP into the persistent
    # registry Stage 4 reads for the AP-distance and mobile-hotspot signals.
    python3 extract_ap_metadata.py "$NEW_PCAP" "$AP_CHUNK_META" >> "$LOG_FILE" 2>&1
    if [ -f "$AP_CHUNK_META" ]; then
        python3 ap_registry.py "$AP_CHUNK_META" "$AP_REGISTRY_PATH" "$WIFI_CHANNEL" >> "$LOG_FILE" 2>&1
        rm -f "$AP_CHUNK_META"
    fi

    if [ -f "$RAW_VMATRIX" ]; then
        python3 2_1_Temporal_Sanitizer.py "$RAW_VMATRIX" "$TDT_MS" >> "$LOG_FILE" 2>&1
        
        if [ -f "$SANITIZED" ]; then
            # STATE_FILE (Stage 3's sliding window) and TRACK_STATE_FILE (Stage 4's
            # position/track continuity) are deliberately separate files -- see
            # config.env -- so that Stage 4 dropping a stale track can never
            # clobber Stage 3's in-progress window data for that same bucket.
            python3 3_Stage3_Localization.py "$SANITIZED" "$STAGE3_OUT" "$STATE_FILE"
            python3 4_Stage4_Inference.py "$STAGE3_OUT" "$TRACK_STATE_FILE"
            rm -f "$RAW_VMATRIX" "$RAW_ANGLES" "$SANITIZED"
        fi
    fi
    rm -f "$NEW_PCAP"
done