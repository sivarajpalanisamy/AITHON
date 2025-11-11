import librosa
import numpy as np
from pydub import AudioSegment

def safe_print(text):
    """Print with proper encoding handling for Windows console"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for Windows console
        print(text.encode('ascii', 'replace').decode('ascii'))

def detect_peaks(audio_path: str, num_peaks: int = 2):
    """
    Detects 1–2 high‑energy segments (emotional peaks)
    based on RMS loudness changes in the voice track.
    """
    safe_print("🔍 Detecting emotional peaks...")
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    # Normalize RMS and find top peaks
    rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))
    top_indices = np.argsort(rms_norm)[-num_peaks:]
    peak_times = np.sort(times[top_indices])

    safe_print("✅ Detected peaks (sec):" + str([round(t, 2) for t in peak_times]))
    return peak_times


def add_emphasis_sfx(audio_path: str, sfx_path: str, output_path: str, peak_times):
    """
    Adds a short musical or SFX accent (swells) at detected peaks.
    """
    voice = AudioSegment.from_file(audio_path)
    sfx = AudioSegment.from_file(sfx_path) - 6  # reduce so it blends naturally

    for t in peak_times:
        position_ms = int(t * 1000)
        voice = voice.overlay(sfx, position=position_ms)

    voice.export(output_path, format="mp3")
    safe_print(f"✨ Emphasis added! Output: {output_path}")
    return output_path


if __name__ == "__main__":
    voice_test = "cleaned_output.wav"
    sfx_test = "stingers/swell.mp3"  # a short swell or cymbal file
    output_test = "highlighted_voice.mp3"
    peaks = detect_peaks(voice_test)
    add_emphasis_sfx(voice_test, sfx_test, output_test, peaks)
