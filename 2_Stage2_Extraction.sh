#!/bin/bash
# ==============================================================================
# Module: 2_Stage2_Extraction.sh
# Wi-BFI payload extraction and trigger subsequent stages
# ==============================================================================
source ./config.env

echo "[*] Stage 2: Extraction Dispatcher Active. Watching for chunks..."

# Both events are needed, one per delivery mechanism: tcpdump closes a rotated
# chunk (close_write), while 0_replay_pcap.sh renames one in (moved_to).
inotifywait -m -e close_write,moved_to --format "%w%f" "$WATCH_DIR" | while read -r NEW_PCAP
do
    if [[ "$NEW_PCAP" != *.pcap ]]; then continue; fi
    BASE=$(basename "$NEW_PCAP" .pcap)
    
    RAW_VMATRIX="${WATCH_DIR}/${BASE}_vmatrix.npy"
    RAW_ANGLES="${WATCH_DIR}/${BASE}_angles.npy"
    SANITIZED="${WATCH_DIR}/${BASE}_sanitized.npy"
    AP_CHUNK_META="${WATCH_DIR}/${BASE}_ap_metadata.json"

    python3 "$WIBFI_DIR/main.py" "$NEW_PCAP" "$WIFI_STANDARD" "$MIMO_MODE" "$FALLBACK_CONFIG" "$BANDWIDTH" "$MAX_PACKETS" "$RAW_VMATRIX" "$RAW_ANGLES" >> "$LOG_FILE" 2>&1

    # Same chunk, Beacon frames this time: folds each AP's SSID, transmit power
    # and RSSI into the registry Stage 4 reads for AP distance and hotspot class.
    python3 extract_ap_metadata.py "$NEW_PCAP" "$AP_CHUNK_META" >> "$LOG_FILE" 2>&1
    if [ -f "$AP_CHUNK_META" ]; then
        python3 ap_registry.py "$AP_CHUNK_META" "$AP_REGISTRY_PATH" "$WIFI_CHANNEL" >> "$LOG_FILE" 2>&1
        rm -f "$AP_CHUNK_META"
    fi

    if [ -f "$RAW_VMATRIX" ]; then
        python3 2_1_Temporal_Sanitizer.py "$RAW_VMATRIX" "$TDT_MS" >> "$LOG_FILE" 2>&1
        
        if [ -f "$SANITIZED" ]; then
            # STATE_FILE (Stage 3 window) and TRACK_STATE_FILE (Stage 4 tracks)
            # must stay separate files: Stage 4 dropping a stale track would
            # otherwise clobber Stage 3's window for that same bucket.
            python3 3_Stage3_Localization.py "$SANITIZED" "$STAGE3_OUT" "$STATE_FILE"
            python3 4_Stage4_Inference.py "$STAGE3_OUT" "$TRACK_STATE_FILE"
            rm -f "$RAW_VMATRIX" "$RAW_ANGLES" "$SANITIZED"
        fi
    fi
    rm -f "$NEW_PCAP"
done