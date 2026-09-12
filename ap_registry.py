#!/usr/bin/env python3
"""
Module: ap_registry.py
Merges per-chunk AP metadata (extract_ap_metadata.py's output) into a
persistent, cross-chunk registry, and derives each AP's Monitor-Card range
from its own beacon RSSI -- an independently measured third leg of the
localization triangle (Client-AP bearing from BFI, Client-MonitorCard range
from client_rssi, AP-MonitorCard range from here), replacing the previously
fully hardcoded AP position with one leg actually grounded in a measurement.
The coordinate convention still only needs this distance, not a compass
bearing to the AP: the local/ego-centric frame is defined with the AP on the
positive Y-axis by construction (see 4_Stage4_Inference.py).

Beacon RSSI is smoothed with a simple exponential moving average across
chunks (BEACON_RSSI_EMA_ALPHA) rather than used raw per-chunk, so the
derived AP distance doesn't jitter chunk-to-chunk the way a single
CHUNK_TIME-second sample would; the smoothing constant is a plain low-pass
filter, not tied to any particular estimator in the literature.
"""
import sys
import json
import os
from importlib import import_module

tx_power_estimator = import_module('tx_power_estimator')

BEACON_RSSI_EMA_ALPHA = 0.3  # weight given to each new chunk's beacon RSSI mean


def load_registry(registry_path):
    if not os.path.exists(registry_path):
        return {}
    with open(registry_path) as f:
        return json.load(f)


def save_registry(registry_path, registry):
    with open(registry_path, "w") as f:
        json.dump(registry, f)


def _new_entry():
    return {
        "ssid": None,
        "tx_power_dbm": None,
        "tx_power_source": None,
        "beacon_rssi_ema": None,
        "ap_distance_m": None,
    }


def merge_chunk(registry, chunk_metadata, freq_mhz):
    """
    Folds one chunk's extract_ap_metadata.py output into the persistent
    registry, updating each AP's smoothed beacon RSSI and re-deriving its
    Monitor-Card distance from that RSSI plus its estimated transmit power.
    Mutates and returns registry.
    """
    for ap_mac, data in chunk_metadata.items():
        entry = registry.setdefault(ap_mac, _new_entry())

        if data.get("ssid"):
            entry["ssid"] = data["ssid"]

        if data.get("tx_power_dbm") is not None:
            entry["tx_power_dbm"] = data["tx_power_dbm"]
            entry["tx_power_source"] = "MEASURED"

        chunk_rssi = data.get("beacon_rssi_mean")
        if chunk_rssi is not None:
            if entry["beacon_rssi_ema"] is None:
                entry["beacon_rssi_ema"] = chunk_rssi
            else:
                entry["beacon_rssi_ema"] = (
                    (1.0 - BEACON_RSSI_EMA_ALPHA) * entry["beacon_rssi_ema"]
                    + BEACON_RSSI_EMA_ALPHA * chunk_rssi
                )

        if entry["beacon_rssi_ema"] is not None:
            if entry["tx_power_dbm"] is not None:
                tx_power_dbm = entry["tx_power_dbm"]
            else:
                tx_power_dbm, _ = tx_power_estimator.estimate_ap_tx_power_dbm(ap_mac, freq_mhz)
            entry["ap_distance_m"] = tx_power_estimator.fspl_distance_m(
                entry["beacon_rssi_ema"], tx_power_dbm, freq_mhz
            )

    return registry


if __name__ == "__main__":
    # CLI glue for 2_Stage2_Extraction.sh: merge one chunk's freshly extracted
    # AP metadata (extract_ap_metadata.py's output) into the persistent
    # registry and save it back out.
    chunk_metadata_path, registry_path, wifi_channel = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(chunk_metadata_path) as f:
        chunk_metadata = json.load(f)
    registry = load_registry(registry_path)
    merge_chunk(registry, chunk_metadata, tx_power_estimator.channel_to_frequency(wifi_channel))
    save_registry(registry_path, registry)
