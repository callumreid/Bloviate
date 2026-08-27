"""Whisper-acoustics gate.

Answers one question about a clip: is this *true whisper* (no vocal-fold
vibration, hence no fundamental frequency) or *phonated* speech (breathy or
otherwise), which still carries an F0? This matters because TTS voice clones
sold as "whisper" voices routinely produce breathy, often high-pitched
*phonation* rather than true whisper -- and an STT model will score better on
that than on a real whispered utterance, so a ranking built on such a clone is
optimistic and can mis-order models.

Run it on any clip before trusting a whisper eval built on it:

    venv/bin/python eval/validate_acoustics.py path/to/clip.wav [more.wav ...]

Needs praat-parselmouth (`pip install praat-parselmouth`).
"""

import sys

import numpy as np
import parselmouth
from parselmouth.praat import call


def analyze(path: str) -> dict:
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]

    # Restrict the voiced-fraction to speech-energy frames so leading/trailing
    # silence does not dilute it.
    intensity = snd.to_intensity(time_step=0.01)
    ivals = intensity.values[0]
    m = min(len(f0), len(ivals))
    speech = ivals[:m] > (ivals.max() - 25)  # within 25 dB of peak
    f0m = f0[:m]

    voiced_all = int(np.sum(f0 > 0)) / max(len(f0), 1)
    voiced_speech = int(np.sum((f0m > 0) & speech)) / max(int(speech.sum()), 1)

    harm = call(snd, "To Harmonicity (cc)", 0.01, 60, 0.1, 1.0)
    hnr = call(harm, "Get mean", 0, 0)
    voiced_f0 = f0[f0 > 0]

    return {
        "dur_s": round(snd.duration, 2),
        "voiced_fraction_speech": round(voiced_speech, 3),
        "voiced_fraction_all": round(voiced_all, 3),
        "mean_f0_hz": round(float(voiced_f0.mean()), 1) if len(voiced_f0) else None,
        "mean_hnr_db": round(float(hnr), 2),
        "rms": round(float(np.sqrt(np.mean(snd.values ** 2))), 5),
    }


def verdict(voiced_fraction_speech: float) -> str:
    if voiced_fraction_speech < 0.15:
        return "TRUE-WHISPER-LIKE (no meaningful pitch -- good whisper proxy)"
    if voiced_fraction_speech < 0.45:
        return "PARTIALLY PHONATED (breathy; optimistic whisper proxy)"
    return "PHONATED (normal/breathy voice, NOT whisper)"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        r = analyze(path)
        print(f"{path}\n  {r}\n  -> {verdict(r['voiced_fraction_speech'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
