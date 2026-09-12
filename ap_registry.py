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
import numpy as np
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
        "beacon_power_ema_mw": None,
        "beacon_rssi_ema": None,
        "ap_distance_m": None,
    }


def merge_chunk(registry, chunk_metadata, freq_mhz, path_loss_exponent=tx_power_estimator.DEFAULT_PATH_LOSS_EXPONENT):
    """
    Folds one chunk's extract_ap_metadata.py output into the persistent
    registry, updating each AP's smoothed beacon RSSI and re-deriving its
    Monitor-Card distance from that RSSI plus its estimated transmit power.
    Mutates and returns registry.

    The EMA runs in the linear (mW) domain -- stored as beacon_power_ema_mw,
    converted to dB only when needed -- for the same reason
    tx_power_estimator.mean_rssi_dbm averages within a chunk that way: each
    chunk's beacon_rssi_mean is already a correctly-debiased dB figure, but
    smoothing a *sequence* of dB values across chunks with a plain
    dB-domain EMA reintroduces the same Jensen's-inequality bias one level
    up, understating the true average power across chunks.
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
            chunk_power_mw = 10 ** (chunk_rssi / 10.0)
            if entry.get("beacon_power_ema_mw") is None:
                entry["beacon_power_ema_mw"] = chunk_power_mw
            else:
                entry["beacon_power_ema_mw"] = (
                    (1.0 - BEACON_RSSI_EMA_ALPHA) * entry["beacon_power_ema_mw"]
                    + BEACON_RSSI_EMA_ALPHA * chunk_power_mw
                )
            entry["beacon_rssi_ema"] = float(10.0 * np.log10(entry["beacon_power_ema_mw"]))

        if entry["beacon_rssi_ema"] is not None:
            if entry["tx_power_dbm"] is not None:
                tx_power_dbm = entry["tx_power_dbm"]
            else:
                tx_power_dbm, _ = tx_power_estimator.estimate_ap_tx_power_dbm(ap_mac, freq_mhz)
            entry["ap_distance_m"] = tx_power_estimator.log_distance_m(
                entry["beacon_rssi_ema"], tx_power_dbm, freq_mhz, path_loss_exponent
            )

    return registry


if __name__ == "__main__":
    # CLI glue for 2_Stage2_Extraction.sh: merge one chunk's freshly extracted
    # AP metadata (extract_ap_metadata.py's output) into the persistent
    # registry and save it back out.
    chunk_metadata_path, registry_path, wifi_channel = sys.argv[1], sys.argv[2], int(sys.argv[3])
    path_loss_exponent = float(os.getenv("PATH_LOSS_EXPONENT", tx_power_estimator.DEFAULT_PATH_LOSS_EXPONENT))
    with open(chunk_metadata_path) as f:
        chunk_metadata = json.load(f)
    registry = load_registry(registry_path)
    merge_chunk(
        registry, chunk_metadata, tx_power_estimator.channel_to_frequency(wifi_channel), path_loss_exponent
    )
    save_registry(registry_path, registry)
