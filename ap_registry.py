#!/usr/bin/env python3
"""
Module: ap_registry.py
Merge per-chunk AP metadata (extract_ap_metadata.py) into a persistent
registry and derive each AP's range from its own beacon RSSI.

That range is the third leg of the localization triangle: client-AP bearing
from BFI, client-monitor range from client_rssi, AP-monitor range from here.
Only the distance is needed, not a bearing -- the frame puts the AP on the
positive Y-axis by construction (4_Stage4_Inference.py).

Beacon RSSI is smoothed across chunks (BEACON_RSSI_EMA_ALPHA) so the derived
distance does not jitter on a single chunk's sample. Plain low-pass, not tied
to any published estimator.
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
    Fold one chunk's metadata into the registry, updating each AP's smoothed
    beacon RSSI and re-deriving its distance. Mutates and returns registry.

    The EMA runs in the linear (mW) domain for the same reason
    tx_power_estimator.mean_rssi_dbm does: each chunk's beacon_rssi_mean is
    already debiased, but smoothing a sequence of dB values in the dB domain
    reintroduces the same Jensen bias one level up.
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
