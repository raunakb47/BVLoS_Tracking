#!/bin/bash
# ==============================================================================
# Module: 2_Stage2_Extraction.sh
# Watch WATCH_DIR and run Stage 2 then Stage 3 on each chunk that lands.
# ==============================================================================
source ./config.env

SESSION_DIR="${SESSION_DIR:-./session_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SESSION_DIR/stage1" "$SESSION_DIR/stage2" "$SESSION_DIR/stage3"

LOG_PREFIX="$SESSION_DIR/stage2/observe"
SOLVE_OUT="$SESSION_DIR/stage3/solve.jsonl"
export SOLVE_OUT
TIMING_LOG="$SESSION_DIR/timing.jsonl"
PIPELINE_LOG="$SESSION_DIR/pipeline.log"
export TIMING_LOG
export SOLVE_REPORT="$SESSION_DIR/stage3/solve.txt"

echo "[*] Stage 2/3 watcher on $WATCH_DIR -> $SESSION_DIR"

# Both events are needed, one per delivery mechanism: tcpdump closes a rotated
# chunk (close_write), while 0_replay_pcap.sh renames one in (moved_to).
inotifywait -m -e close_write,moved_to --format "%w%f" "$WATCH_DIR" | while read -r NEW_PCAP
do
    if [[ "$NEW_PCAP" != *.pcap ]]; then continue; fi
    CHUNK=$(basename "$NEW_PCAP")
    ARRIVED=$(date +%s.%N)

    # Stage 1's output for this chunk, kept so a run can be re-read end to end.
    cp "$NEW_PCAP" "$SESSION_DIR/stage1/$CHUNK"

    # Stage 2: frames -> appended observable log. Wi-BFI is invoked as a
    # subprocess by observe.py, so the extractor stays a standalone tool.
    python3 observe.py "$NEW_PCAP" "$LOG_PREFIX" "$WIBFI_DIR" "$TIMING_LOG" \
        2>>"$PIPELINE_LOG" | tee -a "$PIPELINE_LOG"

    # Stage 3: log -> bearings. Re-solved over the whole log each time, since
    # the log is the only state and a later chunk changes earlier buckets.
    python3 dispatch.py "$LOG_PREFIX" "${SITE_JSON:--}" \
        2>>"$PIPELINE_LOG" | grep '^\[stage3\]' | tee -a "$PIPELINE_LOG"

    DONE=$(date +%s.%N)
    CHAIN=$(echo "$DONE $ARRIVED" | awk '{printf "%.0f", ($1-$2)*1000}')
    echo "[chain ] $CHUNK  arrival to stage3 complete: ${CHAIN} ms" | tee -a "$PIPELINE_LOG"
    echo "{\"stage\":\"chain\",\"chunk\":\"$CHUNK\",\"ms\":$CHAIN}" >> "$TIMING_LOG"

    rm -f "$NEW_PCAP"
done
