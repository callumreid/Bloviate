"""Degrade a clean clip into the conditions Bloviate actually sees.

Given a clean studio/recorded clip, produce versions that mimic the runtime
audio path so an STT comparison reflects reality rather than pristine input:

  clean       resampled 16 kHz mono, no processing (upper bound)
  quiet       scaled to a realistic whisper level, no noise
  office      whisper level + speech-shaped office babble at a given SNR
  office_app  office + Bloviate's NoiseSuppressor (stationary spectral gate)
              + the prerecorded AGC -- i.e. what the cloud final pass receives
  office_app_nogate  same but with the spectral gate bypassed (the A/B)

The point of the last two is to isolate the `noisereduce` stationary gate:
`office_app_nogate` keeps the AGC but drops the gate. Comparing the two tells
you whether the gate is helping or (as measured on a whisper voice clone in
Aug 2026) quietly hurting.

Levels are chosen to match logged real whisper input (raw_rms ~0.01-0.02).
"""

import numpy as np
from scipy import signal as sig
import soundfile as sf

# Bloviate's NoiseSuppressor lives in the app source tree.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TARGET_SR = 16000
WHISPER_RMS = 0.012          # matches logged real whisper raw_rms 0.010-0.022
DEFAULT_SNR_DB = 15.0


def load_16k(path: str) -> np.ndarray:
    d, sr = sf.read(path, dtype="float64")
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != TARGET_SR:
        g = np.gcd(sr, TARGET_SR)
        d = sig.resample_poly(d, TARGET_SR // g, sr // g)
    return d


def _scale_rms(d, target):
    return d * (target / max(np.sqrt(np.mean(d ** 2)), 1e-9))


def _office_noise(n, seed):
    """White noise shaped into speech-band babble with slow AM and sparse clicks."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n + TARGET_SR)
    b, a = sig.butter(2, [200 / (TARGET_SR / 2), 4000 / (TARGET_SR / 2)], "band")
    shaped = sig.lfilter(b, a, white)
    b2, a2 = sig.butter(1, 800 / (TARGET_SR / 2))
    shaped = 0.6 * sig.lfilter(b2, a2, shaped) + 0.4 * shaped
    t = np.arange(len(shaped)) / TARGET_SR
    shaped *= 1.0 + 0.5 * np.sin(2 * np.pi * 3.1 * t + rng.uniform(0, 6)) * np.sin(2 * np.pi * 0.7 * t)
    for _ in range(max(2, n // TARGET_SR)):
        i = rng.integers(0, n)
        k = min(200, n + TARGET_SR - i)
        shaped[i:i + k] += rng.standard_normal(k) * np.hanning(k) * 2.5
    return shaped[:n]


def _add_noise(d, snr_db, seed):
    noise = _office_noise(len(d), seed)
    noise *= np.sqrt(np.mean(d ** 2) / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return d + noise


def _bloviate_gate(d, prop_decrease):
    """Bloviate's stationary spectral gate (noisereduce), at a given strength."""
    if prop_decrease <= 0:
        return d
    import noisereduce as nr
    return nr.reduce_noise(
        y=d.astype("float32"), sr=TARGET_SR, stationary=True,
        prop_decrease=prop_decrease, thresh_n_mult_nonstationary=2,
    ).astype("float64")


def _prerecorded_agc(d):
    """Mirror transcriber prerecorded gain: target_rms 0.05, max +45 dB, ceiling 0.95."""
    gain = min(0.05 / max(np.sqrt(np.mean(d ** 2)), 1e-9), 10 ** (45 / 20))
    out = d * gain
    peak = np.abs(out).max()
    if peak > 0.95:
        out *= 0.95 / peak
    return out


def make_condition(clean, condition, snr_db=DEFAULT_SNR_DB, seed=1,
                   gate_strength=0.7):
    """Return one condition as a 16 kHz float array. `clean` is a 16 kHz array."""
    if condition == "clean":
        return clean
    if condition == "quiet":
        return _scale_rms(clean, WHISPER_RMS)
    noisy = _add_noise(_scale_rms(clean, WHISPER_RMS), snr_db, seed)
    if condition == "office":
        return noisy
    if condition == "office_app":
        return _prerecorded_agc(_bloviate_gate(noisy, gate_strength))
    if condition == "office_app_nogate":
        return _prerecorded_agc(noisy)
    raise ValueError(f"unknown condition {condition!r}")


def write_condition(clean_path, out_path, condition, **kw):
    out = make_condition(load_16k(clean_path), condition, **kw)
    sf.write(out_path, np.clip(out, -1, 1).astype("float32"), TARGET_SR, subtype="PCM_16")
    return out_path
