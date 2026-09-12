#!/usr/bin/env python3
"""
Module: tx_power_estimator.py
Transmit-power estimates and the shared log-distance ranging formula.

Two transmitters need separate defaults. client_rssi is measured on the
client's own Compressed Beamforming Report, so ranging it needs the client's
transmit power; ranging an overheard Beacon needs the AP's. Clients typically
transmit well below AP power.

Transmit power cannot be recovered from RSSI alone: in
P_r = P_t + G_t + G_r - PathLoss(d), P_t and PathLoss(d) enter only through
their difference, so one link cannot separate them (Wang & Ho, "Blind Received
Signal Strength Difference Based Source Localization With System Parameter
Errors," IEEE Trans. Signal Processing). Per-packet BFI SNR adds another
unknown-distance link, not a second equation.

What is recoverable is what a transmitter already broadcasts. An AP's
Country / Power Constraint / Transmit Power Envelope elements ride in every
Beacon and Probe Response, in plaintext, with no association needed -- hence
the MEASURED tier for APs. A client has no continuous equivalent: the nearest
analog, the Power Capability element, appears once in its Association Request,
which this pipeline is unlikely to have been listening for. Client ranging
therefore has no MEASURED tier.
"""
import json
import numpy as np


def channel_to_frequency(channel):
    """
    IEEE 802.11 channel number to center frequency (MHz), 2.4 and 5 GHz.
    Shared by the client-ranging and AP-ranging paths so both use one mapping.
    """
    if 1 <= channel <= 13:
        return 2407.0 + (5.0 * channel)
    elif channel == 14:
        return 2484.0
    elif 32 <= channel <= 177:
        return 5000.0 + (5.0 * channel)
    else:
        # Fallback to standard Ch 36 if an unsupported channel is provided
        return 5180.0


# ETSI EN 300 328 / EN 301 893 mean-EIRP limits, which also match the modal
# configured power of commodity APs. FCC 47 CFR 15.247/15.407 allow 30-36 dBm
# depending on band, so these are mid-range, not worst-case, estimates.
DEFAULT_TX_POWER_DBM_AP_2G4 = 20.0
DEFAULT_TX_POWER_DBM_AP_5G = 23.0

# Approximate figure for battery-powered clients (~15 mW), not a regulatory
# constant like the AP values above, so a rougher estimate. No band split: no
# equally solid band-specific source was found for client devices.
DEFAULT_TX_POWER_DBM_CLIENT = 12.0


def _normalize_mac(mac_address):
    return mac_address.strip().lower()


def load_tx_power_table(json_path):
    """
    Load a {mac_address: tx_power_dbm} table, the format written by
    extract_ap_metadata.py, for estimate_ap_tx_power_dbm()'s tx_power_table.
    """
    with open(json_path) as f:
        raw = json.load(f)
    return {_normalize_mac(mac): float(dbm) for mac, dbm in raw.items()}


def estimate_ap_tx_power_dbm(ap_mac, freq_mhz, tx_power_table=None):
    """
    Return (tx_power_dbm, source) for ranging on an AP's Beacon/Probe Response
    RSSI. source is "MEASURED" when the AP's own advertised elements supplied
    the value, "DEFAULT" for the band constant.

    A MEASURED value is the AP's advertised regulatory maximum, not its
    instantaneous power if it runs 802.11h Transmit Power Control below that
    ceiling. Better than a band-wide guess, not a per-packet measurement.
    """
    if tx_power_table:
        measured = tx_power_table.get(_normalize_mac(ap_mac))
        if measured is not None:
            return measured, "MEASURED"

    default_dbm = DEFAULT_TX_POWER_DBM_AP_5G if freq_mhz > 3000 else DEFAULT_TX_POWER_DBM_AP_2G4
    return default_dbm, "DEFAULT"


def estimate_client_tx_power_dbm():
    """
    Return (tx_power_dbm, source) for ranging on client_rssi. Always
    "DEFAULT": see the module docstring for why clients have no MEASURED tier.
    """
    return DEFAULT_TX_POWER_DBM_CLIENT, "DEFAULT"


DEFAULT_PATH_LOSS_EXPONENT = 2.0  # 2.0 = free space (FSPL); indoor NLOS is typically higher, see below


def mean_rssi_dbm(rssi_samples_dbm):
    """
    Average dBm readings in the linear (mW) domain and convert back.

    dB is a concave transform of power, so by Jensen's inequality a mean of dB
    readings sits at or below the dB of the mean power. On a fading signal that
    understatement reads as extra path loss and inflates the range estimate.
    """
    rssi_samples_dbm = np.asarray(rssi_samples_dbm, dtype=float)
    linear_mw = 10 ** (rssi_samples_dbm / 10.0)
    return float(10.0 * np.log10(np.mean(linear_mw)))


def log_distance_m(rssi_dbm, tx_power_dbm, freq_mhz, path_loss_exponent=DEFAULT_PATH_LOSS_EXPONENT):
    """
    Distance (m) from the log-distance path loss model:
        PL(d) = PL(1m) + 10 n log10(d),  PL(1m) = 20 log10(freq_mhz) - 27.55
    Path loss in dB is transmit power minus received power, ignoring antenna
    gains, which are unknown for a device this pipeline never associates with.
    Shared by the client-ranging and AP-ranging paths.

    n=2 is free space and reduces this to plain FSPL. Indoor NLOS measures well
    above that (ITU-R P.1238-style calibrations report n=2.83 @2.4GHz, n=3.89
    @5.3GHz), so assuming n=2 through walls OVERESTIMATES distance: wall loss
    has no other explanation in the model than more distance. Numerically, a
    source 10 m away in an n=3 environment reads as ~32 m under n=2, and as
    10 m under n=3. n is exposed as PATH_LOSS_EXPONENT because neither this
    pipeline nor the literature can supply the right n for an unsurveyed room.
    """
    path_loss_db = tx_power_dbm - rssi_dbm
    path_loss_1m_db = (20 * np.log10(freq_mhz)) - 27.55
    return float(10 ** ((path_loss_db - path_loss_1m_db) / (10.0 * path_loss_exponent)))


def range_uncertainty_m(distance_m, rssi_std_db, path_loss_exponent=DEFAULT_PATH_LOSS_EXPONENT):
    """
    Delta-method propagation of RSSI spread into distance uncertainty.
    Differentiating log_distance_m gives
        d(distance)/d(rssi) = -distance * ln(10) / (10 n)
    so an RSSI standard deviation of rssi_std_db contributes roughly
    distance * ln(10) / (10 n) * rssi_std_db metres. Standard first-order error
    propagation, not a literature-specific estimator.
    """
    return float(distance_m * np.log(10.0) / (10.0 * path_loss_exponent) * rssi_std_db)
