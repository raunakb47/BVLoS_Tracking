#!/bin/bash
source ./config.env

WATCH_DIR="./live_chunks"
WIBFI_DIR="../Wi-BFI" 

echo "[*] Extraction daemon active. Monitoring spool directory..."
echo "    -> Operational Mode: Agnostic Full-Spectrum Extraction"

inotifywait -m -e close_write --format "%w%f" $WATCH_DIR | while read NEW_PCAP
do
    BASE_NAME=$(basename "$NEW_PCAP" .pcap)
    
    # Execute Wi-BFI without any MAC filtering
    python $WIBFI_DIR/main.py \
        "$NEW_PCAP" \
        "$WIFI_STANDARD" \
        "$MIMO_MODE" \
        "$FALLBACK_CONFIG" \
        "$BANDWIDTH" \
        "$MAX_PACKETS" \
        "vmatrix_batch_${BASE_NAME}" \
        "angles_batch_${BASE_NAME}" > /dev/null 2>&1
    
    if [ -f "vmatrix_batch_${BASE_NAME}.npy" ]; then
        python 3_bvlos_tracker.py "vmatrix_batch_${BASE_NAME}.npy"
    fi
done