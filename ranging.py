#!/usr/bin/env python3
"""
Module: ranging.py
Stage 4. Largest distance from the monitor to a transmitter that a received
power allows.

Log-distance model with the free-space loss at 1 m as reference:

    RSSI = P_tx - PL(1 m) - 10 n log10(d),   PL(1 m) = 20 log10(f_MHz) - 27.55

P_tx and n are unknown for a device this framework never associates with, and
one link cannot separate them from distance. The distance is largest with P_tx
at its legal ceiling and n at its minimum, so that pair gives d_max and the
range is a disk: the transmitter is no farther than d_max. Neither a power
floor nor an upper exponent is assumed, so no inner radius is claimed.

The ceiling comes from data, first source found (power_ceiling()):
  1. the Country element of the transmitter's own AP (Beacon, Probe Response)
  2. any Country element heard in the same band: the regulatory domain is a
     property of the site, not of one AP
  3. the monitor host's regulatory domain (iw reg get, logged by the watcher)
Absent all three the range is unbounded and reported as None.

Antenna gains are folded into P_tx: the limits are EIRP.
"""
import os
import re

import numpy as np

# 802.11 band edges, MHz. A Country element heard in one band says nothing
# about another's limits.
BANDS_MHZ = ((2400.0, 2500.0), (5150.0, 5925.0), (5925.0, 7125.0))

_RULE = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*@[^)]*\),\s*"
                   r"\([^,]*,\s*(\d+(?:\.\d+)?)\s*(mBm)?\s*\)")


def exponent_min(environ=None):
    """PATH_LOSS_EXPONENT_MIN from config.env; raises unless a positive number."""
    environ = os.environ if environ is None else environ
    value = environ.get("PATH_LOSS_EXPONENT_MIN")
    try:
        exponent = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"PATH_LOSS_EXPONENT_MIN={value!r} is not a number") from None
    if not exponent > 0:
        raise ValueError(f"PATH_LOSS_EXPONENT_MIN={value!r} is not positive")
    return exponent


def band_of(freq_mhz):
    """Index into BANDS_MHZ, or None outside every band."""
    if freq_mhz is None:
        return None
    for index, (low, high) in enumerate(BANDS_MHZ):
        if low <= freq_mhz < high:
            return index
    return None


def parse_regdomain(text):
    """
    Frequency rules [(start MHz, end MHz, max EIRP dBm)] from `iw reg get`
    output. Only the global section is read; per-phy sections follow it.
    """
    rules = []
    for line in text.splitlines():
        if line.startswith("phy#"):
            break
        match = _RULE.search(line)
        if match:
            eirp = float(match.group(3)) / (100.0 if match.group(4) else 1.0)
            rules.append((float(match.group(1)), float(match.group(2)), eirp))
    return rules


def load_regdomain(path):
    """Rules from a logged `iw reg get`; empty when the file is absent."""
    if not path or not os.path.exists(path):
        return []
    with open(path) as handle:
        return parse_regdomain(handle.read())


def power_ceiling(freq_mhz, beamformer, ap_records, regdomain):
    """
    (ceiling dBm, source) for a transmitter heard at freq_mhz whose AP is
    beamformer (an AP ranging itself passes its own address). source is
    "own_ap", "band", "host_regdomain" or None with a None ceiling.
    """
    band = band_of(freq_mhz)
    own = [r["tx_power_dbm"] for r in ap_records
           if r.get("tx_power_dbm") is not None and beamformer in r.get("beamformers", ())]
    if own:
        return own[-1], "own_ap"
    heard = [r["tx_power_dbm"] for r in ap_records
             if r.get("tx_power_dbm") is not None and band is not None
             and band_of(r.get("freq_mhz")) == band]
    if heard:
        return max(heard), "band"
    if freq_mhz is not None:
        covering = [eirp for start, end, eirp in regdomain if start <= freq_mhz <= end]
        if covering:
            return max(covering), "host_regdomain"
    return None, None


def free_space_loss_1m_db(freq_mhz):
    """Friis free-space loss at 1 m, dB."""
    return 20.0 * np.log10(freq_mhz) - 27.55


def max_range_m(rssi_dbm, ceiling_dbm, freq_mhz, exponent):
    """
    Largest distance consistent with rssi_dbm, or None when any input is
    missing. A reading above what the ceiling allows at 1 m bounds the
    distance by 1 m, the model's reference: every exponent then gives less.
    """
    if rssi_dbm is None or ceiling_dbm is None or not freq_mhz:
        return None
    excess_db = ceiling_dbm - rssi_dbm - free_space_loss_1m_db(freq_mhz)
    if excess_db <= 0:
        return 1.0
    return float(10.0 ** (excess_db / (10.0 * exponent)))


def client_range(output, facts, ap_records, regdomain, exponent):
    """{"max_m", "ceiling_dbm", "ceiling_source"} for a Stage 3 output's client."""
    summary = output.get("client_rssi")
    freq = facts.get("centre_freq_mhz")
    ceiling, source = power_ceiling(freq, facts["beamformer"], ap_records, regdomain)
    return {"max_m": max_range_m(summary and summary["median_dbm"], ceiling, freq, exponent),
            "ceiling_dbm": ceiling, "ceiling_source": source}


def ap_range(ap_record, ap_records, regdomain, exponent):
    """{"max_m", "ceiling_dbm", "ceiling_source"} for a Stage 3 "ap" record."""
    summary = ap_record.get("rssi")
    freq = ap_record.get("freq_mhz")
    ceiling, source = power_ceiling(freq, ap_record["bssid"],
                                    [dict(ap_record, beamformers=[ap_record["bssid"]])]
                                    + list(ap_records), regdomain)
    return {"max_m": max_range_m(summary and summary["median_dbm"], ceiling, freq, exponent),
            "ceiling_dbm": ceiling, "ceiling_source": source}
