import streamlit as st
import numpy as np
import json
import io
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy import signal
import matplotlib.pyplot as plt

st.set_page_config(page_title="Audio Analyze & Recommend", layout="wide")

def load_mono_bytes(file_bytes, sr_target=48000):
    # file_bytes: bytes-like object
    # use librosa to load from bytes
    with io.BytesIO(file_bytes) as f:
        y, sr = librosa.load(f, sr=sr_target, mono=True)
    return y, sr

def integrated_loudness(y, sr):
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(y)

def spectral_centroid(y, sr):
    return float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())

def spectral_slope(y, sr):
    S = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mags = np.mean(20*np.log10(S+1e-12), axis=1)
    p = np.polyfit(freqs, mags, 1)
    return float(p[0]), float(p[1])

def find_tonal_peaks(y, sr, n_fft=4096):
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    avg = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peaks, props = signal.find_peaks(avg, height=np.max(avg)*0.06, distance=5)
    peak_freqs = freqs[peaks]
    return peak_freqs[:10].tolist()

def eq_match_suggestion(y_target, y_ref, sr):
    n_fft = 4096
    S_t = np.mean(np.abs(librosa.stft(y_target, n_fft=n_fft)), axis=1)
    S_r = np.mean(np.abs(librosa.stft(y_ref, n_fft=n_fft)), axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    diff_db = 20*np.log10((S_r+1e-12)/(S_t+1e-12))
    bands = [80, 160, 300, 600, 1200, 3000, 6000, 12000]
    gains = {}
    for b in bands:
        idx = np.argmin(np.abs(freqs - b))
        gains[str(b)] = float(np.clip(diff_db[idx], -6.0, 6.0))
    return gains

def recommend_params(l_in, l_cl, centroid_in, centroid_cl, peak_freqs):
    rec = {}
    rec['target_lufs'] = -16.0
    rec['gain_adjust_db'] = float(l_cl - l_in)
    rec['highpass_hz'] = 80
    notch = None
    for f in peak_freqs:
        if abs(f - 50) < 3 or abs(f - 60) < 3:
            notch = int(round(f))
            break
    rec['notch_hz'] = notch
    rec['notch_q'] = 8 if notch else None
    if centroid_cl - centroid_in > 200:
        rec['presence_boost'] = {"freq": 3500, "gain_db": 2.5}
    else:
        rec['presence_boost'] = {"freq": 3500, "gain_db": 2.0}
    rec['denoiser'] = "RNNoise (fast, CPU) -> DeepFilterNet/VoiceFixer if higher quality needed"
    rec['dereverb'] = "WPE (nara_wpe) conservative taps=10, delay=3, iterations=2-3"
    rec['compressor'] = {"ratio": "2.5:1", "attack_ms": 6, "release_ms": 150, "threshold_db": -18}
    rec['deesser'] = {"freq": 5500, "threshold_db": -30, "ratio": "4:1"}
    rec['true_peak_max_dbtp'] = -1.0
    rec['duck_reduction_db'] = -8.0
    return rec

st.title("Audio Analyze & Recommendation (Input vs Clean Reference)")
st.markdown("Upload the **raw input** and the **expected cleaned output**. The app will compute loudness, spectral stats, likely tonal peaks, a simple EQ-match suggestion, and a recommended processing recipe.")

col1, col2 = st.columns(2)
with col1:
    in_file = st.file_uploader("Upload RAW input audio", type=["wav","mp3","m4a","flac"], key="input")
with col2:
    ref_file = st.file_uploader("Upload CLEAN reference audio", type=["wav","mp3","m4a","flac"], key="ref")

run_btn = st.button("Analyze and Recommend")

if run_btn:
    if in_file is None or ref_file is None:
        st.warning("Please upload both files.")
    else:
        try:
            with st.spinner("Loading audio and computing metrics..."):
                in_bytes = in_file.read()
                ref_bytes = ref_file.read()
                y_in, sr = load_mono_bytes(in_bytes, sr_target=48000)
                y_cl, sr2 = load_mono_bytes(ref_bytes, sr_target=48000)
                assert sr == sr2

                l_in = integrated_loudness(y_in, sr)
                l_cl = integrated_loudness(y_cl, sr)
                cent_in = spectral_centroid(y_in, sr)
                cent_cl = spectral_centroid(y_cl, sr)
                slope_in, _ = spectral_slope(y_in, sr)
                slope_cl, _ = spectral_slope(y_cl, sr)
                peaks = find_tonal_peaks(y_in, sr)
                eq_gains = eq_match_suggestion(y_in, y_cl, sr)
                rec = recommend_params(l_in, l_cl, cent_in, cent_cl, peaks)

                out = {
                    "sr": int(sr),
                    "loudness_input_lufs": float(l_in),
                    "loudness_clean_lufs": float(l_cl),
                    "spectral_centroid_input_hz": float(cent_in),
                    "spectral_centroid_clean_hz": float(cent_cl),
                    "spectral_slope_input_db_per_hz": float(slope_in),
                    "spectral_slope_clean_db_per_hz": float(slope_cl),
                    "tonal_peaks_hz": peaks,
                    "eq_match_suggest_db_per_band": eq_gains,
                    "recommendations": rec
                }

            st.success("Analysis complete — recommendations ready.")
            # Show JSON
            st.subheader("Summary JSON")
            st.code(json.dumps(out, indent=2), language="json")

            # Download button for JSON
            st.download_button("Download analysis JSON", data=json.dumps(out, indent=2).encode("utf-8"),
                                file_name="analysis_recommendation.json", mime="application/json")

            # Show some visualizations
            st.subheader("Visual comparisons")
            fig, axes = plt.subplots(2, 1, figsize=(10,6))
            times_in = np.linspace(0, len(y_in)/sr, num=len(y_in))
            times_cl = np.linspace(0, len(y_cl)/sr, num=len(y_cl))
            axes[0].plot(times_in, y_in, alpha=0.6, label="Input")
            axes[0].plot(times_cl, y_cl, alpha=0.6, label="Clean reference")
            axes[0].set_title("Waveforms (time domain)")
            axes[0].legend(loc="upper right")
            # Spectral centroid bar
            axes[1].bar([0,1], [cent_in, cent_cl], tick_label=["Input centroid (Hz)","Clean centroid (Hz)"])
            axes[1].set_title("Spectral centroid comparison")
            st.pyplot(fig)

            st.markdown("### Key numeric metrics")
            st.write(f"**Integrated loudness (LUFS)** — Input: **{l_in:.2f} LUFS**, Clean: **{l_cl:.2f} LUFS**")
            st.write(f"**Spectral centroid** — Input: **{cent_in:.1f} Hz**, Clean: **{cent_cl:.1f} Hz**")
            st.write(f"**Detected tonal peaks (likely hum/tones)** — {peaks}")

            st.markdown("### EQ match suggestions (per band)")
            st.table({ "band_hz": list(eq_gains.keys()), "gain_db": list(eq_gains.values()) })

            st.markdown("### Concrete recommended processing steps")
            st.json(rec)

        except Exception as e:
            st.error(f"Processing error: {e}")
            raise
