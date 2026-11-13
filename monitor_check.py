#!/usr/bin/env python3
"""
Module: monitor_check.py
One-time monitor check: the median RSSI the monitor measures from a
transmitter held 1 m away, for MONITOR_CHECK_RSSI_1M_DBM and
MONITOR_CHECK_FREQ_MHZ in config.env (ranging.py).

Procedure: run Stages 1 and 2 with a client associated to a sounding AP, hold
the client 1 m from the monitor's antenna for about a minute, and turn it
through several orientations while it is there, so that the median spans
them. Then pass the session and the client's MAC, with the start and end of
the hold when the session holds anything else.

The readings are the RSSI on the client's beamforming reports, the same
readings a client range is computed from (dispatch.py, rssi_summary()).
"""
import argparse
import os
import sys

import numpy as np

from observe import read_log


def check(log_prefix, mac, start=None, end=None):
    """{median_dbm, n, low_dbm, high_dbm, freq_mhz} over the client's reports, or None."""
    mac = mac.lower()
    rssi, freqs = [], []
    for record in read_log(log_prefix):
        if record.get("kind") != "bfi" or record.get("transmitter", "").lower() != mac:
            continue
        if (start is not None and record["t"] < start) or (end is not None and record["t"] > end):
            continue
        if record.get("rssi") is None:
            continue
        rssi.append(record["rssi"])
        if record.get("freq_mhz"):
            freqs.append(record["freq_mhz"])
    if not rssi:
        return None
    return {"median_dbm": float(np.median(rssi)), "n": len(rssi),
            "low_dbm": float(min(rssi)), "high_dbm": float(max(rssi)),
            "freq_mhz": float(np.median(freqs)) if freqs else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("session_dir")
    parser.add_argument("mac", help="the client's MAC address")
    parser.add_argument("--start", type=float, help="start of the 1 m hold, epoch seconds")
    parser.add_argument("--end", type=float, help="end of the 1 m hold, epoch seconds")
    args = parser.parse_args()
    result = check(os.path.join(args.session_dir, "stage2", "observe"), args.mac,
                   args.start, args.end)
    if result is None:
        print(f"no beamforming reports with RSSI from {args.mac}", file=sys.stderr)
        raise SystemExit(1)
    print(f"{result['n']} reports, median {result['median_dbm']:.1f} dBm "
          f"(from {result['low_dbm']:.0f} to {result['high_dbm']:.0f} dBm)")
    if result["freq_mhz"] is None:
        print("no frequency on the reports; set MONITOR_CHECK_FREQ_MHZ by hand", file=sys.stderr)
        raise SystemExit(1)
    print("config.env:")
    print(f'export MONITOR_CHECK_RSSI_1M_DBM="{result["median_dbm"]:.1f}"')
    print(f'export MONITOR_CHECK_FREQ_MHZ="{result["freq_mhz"]:.0f}"')


if __name__ == "__main__":
    main()
