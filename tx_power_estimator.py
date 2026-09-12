#!/usr/bin/env python3
"""
Module: tx_power_estimator.py
Estimates a transmitter's EIRP for FSPL-based ranging.

The previous FSPL formula (4_Stage4_Inference.py's ray_circle_intersection)
used abs(client_rssi) directly as the total path loss, which implicitly
assumes 0 dBm transmit power. Real Wi-Fi transmitters run well above that, so
every distance the old formula produced was too small by roughly the
unmodeled gap -- 10**(20/20) = 10x too small for a 20 dB gap. That single
missing term, not antenna geometry or wall attenuation, was the dominant
source of error in the ray-circle math (confirmed by direct numerical
testing: realistic indoor RSSI values produced sub-2-meter radii regardless
of true distance).

This module estimates TWO different transmitters, because they need
different defaults: the pipeline's "client_rssi" is the RSSI of the client's
own Compressed Beamforming Report frame (sent client -> AP, overheard at the
monitor card) -- so ranging on it needs the CLIENT's typical transmit power,
not the AP's, and client devices commonly run considerably lower TX power
than APs to conserve battery. Ranging on an overheard AP Beacon needs the
AP's own typical/advertised transmit power instead.

Blindly estimating either device's TX power from received signal strength
alone is not possible without a distance prior -- P_r = P_t + G_t + G_r -
PathLoss(d), and P_t and PathLoss(d) enter only through their difference, so
a single link cannot separate them (confirmed against the RSS-difference/
joint-estimation localization literature, e.g. Wang & Ho, "Blind Received
Signal Strength Difference Based Source Localization With System Parameter
Errors," IEEE Trans. Signal Processing). Per-packet BFI SNR does not change
this: it adds an estimate of received power at the beamformee, but that is
still one more unknown-distance link, not a second independent equation. So
"let the packets do the talking" is implemented here as reading what a
transmitter is *already required to broadcast* -- an AP's standards-mandated
Country/Power Constraint/VHT-HE Transmit Power Envelope elements, carried in
every Beacon and Probe Response, plaintext and sniffable without association
-- with a literature-typical constant as the fallback when that has not been
observed yet. A client has no beacon-equivalent continuous broadcast of its
own transmit power (the nearest standards analog, the Power Capability
element, only appears once in that client's original Association Request,
which the pipeline is very unlikely to have been listening for), so client
ranging currently has no MEASURED tier at all -- DEFAULT is the only source
until/unless that one-time frame is captured.
"""
import json
import numpy as np


def channel_to_frequency(channel):
    """
    Converts IEEE 802.11 channel numbers to center frequency (MHz).
    Supports standard 2.4 GHz and 5 GHz bands. Shared by
    4_Stage4_Inference.py (client-ranging path) and ap_registry.py's CLI
    (AP-ranging path) so both range calculations use the same band mapping.
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
# configured power for commodity APs in practice (FCC 47 CFR 15.247/15.407
# allow more -- up to 30-36 dBm depending on band/sub-band -- so these
# defaults are a mid-of-the-road, not a worst-case-transmit-power, estimate).
DEFAULT_TX_POWER_DBM_AP_2G4 = 20.0
DEFAULT_TX_POWER_DBM_AP_5G = 23.0

# Commonly cited approximate figure for battery-powered client devices
# (~15 mW), not a regulatory constant like the AP figures above -- treat this
# one as a rougher estimate. No band-specific split is used here because no
# equally solid band-specific source was found for client devices.
DEFAULT_TX_POWER_DBM_CLIENT = 12.0


def _normalize_mac(mac_address):
    return mac_address.strip().lower()


def load_tx_power_table(json_path):
    """
    Load a {mac_address: tx_power_dbm} table -- the format written by
    extract_ap_tx_power.py once Stage 1/2 capture Beacon/Probe Response
    frames (see that script) -- for use as the tx_power_table argument to
    estimate_ap_tx_power_dbm().
    """
    with open(json_path) as f:
        raw = json.load(f)
    return {_normalize_mac(mac): float(dbm) for mac, dbm in raw.items()}


def estimate_ap_tx_power_dbm(ap_mac, freq_mhz, tx_power_table=None):
    """
    Returns (tx_power_dbm, source) for an AP-class transmitter (i.e. for
    ranging on an overheard Beacon/Probe Response RSSI). source is
    "MEASURED" when ap_mac has an observed value in tx_power_table (from
    that AP's own advertised Country/Power Constraint/TPE elements), or
    "DEFAULT" when falling back to a literature-typical constant for the
    operating band. A MEASURED value is still the AP's advertised
    *regulatory maximum*, not necessarily its instantaneous transmit power
    if it runs 802.11h Transmit Power Control below that ceiling -- better
    than a band-wide guess, but not a direct per-packet measurement either.
    """
    if tx_power_table:
        measured = tx_power_table.get(_normalize_mac(ap_mac))
        if measured is not None:
            return measured, "MEASURED"

    default_dbm = DEFAULT_TX_POWER_DBM_AP_5G if freq_mhz > 3000 else DEFAULT_TX_POWER_DBM_AP_2G4
    return default_dbm, "DEFAULT"


def estimate_client_tx_power_dbm():
    """
    Returns (tx_power_dbm, source) for ranging on client_rssi (a client's own
    Compressed Beamforming Report frame, overheard at the monitor card).
    Always "DEFAULT" today -- see module docstring for why a MEASURED tier
    isn't available yet for client devices.
    """
    return DEFAULT_TX_POWER_DBM_CLIENT, "DEFAULT"


DEFAULT_PATH_LOSS_EXPONENT = 2.0  # 2.0 = free space (FSPL); indoor NLOS is typically higher, see below


def mean_rssi_dbm(rssi_samples_dbm):
    """
    Averages a set of dBm readings correctly for a fading channel: RSSI in
    dB is a logarithm of received power, and log is a concave function, so
    by Jensen's inequality the arithmetic mean of several dB readings is
    always <= the dB of the mean of the underlying linear powers -- i.e.
    naively averaging dB values systematically understates the true average
    received power on any signal that fades sample to sample (which real
    indoor Wi-Fi RSSI always does). That understatement reads as *extra*
    path loss, which this pipeline's ranging would turn into an
    overestimated distance. Averaging is done here in the linear (mW)
    domain and converted back to dB, which is the textbook-correct way to
    average a fading RSSI series and removes that bias.
    """
    rssi_samples_dbm = np.asarray(rssi_samples_dbm, dtype=float)
    linear_mw = 10 ** (rssi_samples_dbm / 10.0)
    return float(10.0 * np.log10(np.mean(linear_mw)))


def log_distance_m(rssi_dbm, tx_power_dbm, freq_mhz, path_loss_exponent=DEFAULT_PATH_LOSS_EXPONENT):
    """
    Log-distance path loss model distance estimate (meters):
        PL(d) = PL(1m) + 10 * n * log10(d)
    where PL(1m) = 20*log10(freq_mhz) - 27.55 is the free-space loss at a
    1-meter reference (Friis) and n is the path loss exponent -- n=2
    (DEFAULT_PATH_LOSS_EXPONENT, free space) reduces this exactly to the
    FSPL formula this pipeline shipped with originally. Real indoor NLOS
    propagation is measured at n well above 2 -- e.g. an ITU-R P.1238-style
    indoor calibration reporting n=2.83 @2.4GHz / n=3.89 @5.3GHz -- so a
    fixed n=2 assumption (uncorrected free space) systematically
    OVERESTIMATES distance through walls: real wall/obstruction loss beyond
    free-space spreading gets misread as extra distance, since the model has
    no other explanation for the missing signal. Verified numerically: a
    source truly 10m away in a real n=3 environment produces an RSSI that an
    n=2 assumption turns into a ~32m distance estimate, while correctly
    assuming n=3 recovers the true 10m. n is exposed as PATH_LOSS_EXPONENT
    (config.env) specifically so a deployment can calibrate to n=2, keeping
    this framework's "blind, no prior measurement" default, or move to a
    NLOS-representative value once real per-site calibration is available --
    neither this pipeline nor the literature it draws on can tell you the
    correct n for an arbitrary, unsurveyed room in advance.

    Path loss (dB) = transmit power - received power (Friis, ignoring
    antenna gains, which are also unknown for a third-party device this
    pipeline never associates with). Shared by both the client-ranging path
    (4_Stage4_Inference.py's ray_circle_intersection) and the AP-ranging
    path (ap_registry.py) so the same formula isn't maintained in two places.
    """
    path_loss_db = tx_power_dbm - rssi_dbm
    path_loss_1m_db = (20 * np.log10(freq_mhz)) - 27.55
    return float(10 ** ((path_loss_db - path_loss_1m_db) / (10.0 * path_loss_exponent)))


def range_uncertainty_m(distance_m, rssi_std_db, path_loss_exponent=DEFAULT_PATH_LOSS_EXPONENT):
    """
    First-order (delta-method) propagation of RSSI measurement spread into a
    distance uncertainty: differentiating log_distance_m's d = 10^((...)/(10n))
    with respect to rssi_dbm gives
        d(distance)/d(rssi) = -distance * ln(10) / (10 * n)
    so a bucket whose RSSI samples this chunk had standard deviation
    rssi_std_db (already available -- see 3_Stage3_Localization.py's
    windowed RSSI and ap_registry.py's beacon samples) maps to a
    distance-uncertainty contribution of about
        distance * ln(10) / (10 * n) * rssi_std_db
    A noisier RSSI series (deeper fading, more shadowing) now widens the
    reported uncertainty radius instead of the radius only tracking motion/
    mobility as it did previously -- this is standard first-order error
    propagation, not a literature-specific estimator.
    """
    return float(distance_m * np.log(10.0) / (10.0 * path_loss_exponent) * rssi_std_db)
