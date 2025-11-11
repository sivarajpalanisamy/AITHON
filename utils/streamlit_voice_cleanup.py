import streamlit as st
import librosa, soundfile as sf, numpy as np, pyloudnorm as pyln
from nara_wpe.wpe import wpe
import tempfile, os, io, subprocess, traceback, torch

# -------------------------------------------------------------
#  DeepFilterNet Model Loader (cached)
# -------------------------------------------------------------
@st.cache_resource
def load_df_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from df import init_df
        model, df_state, _ = init_df(device=device)
        return model, df_state, device, None
    except Exception as e:
        return None, None, "cpu", str(e)

# -------------------------------------------------------------
#  Utility helpers
# -------------------------------------------------------------
def load_audio(file_bytes, sr_target=48000):
    with io.BytesIO(file_bytes) as f:
        y, sr = librosa.load(f, sr=sr_target, mono=True)
    return y, sr

def save_temp_audio(y, sr):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, y, sr)
    return tmp.name

def denoise_deepfilternet(in_path, out_path, model=None, df_state=None, device="cpu"):
    """
    Denoise using DeepFilterNet if available; fallback to spectral gating otherwise.
    """
    try:
        if model is not None and df_state is not None:
            from df import enhance
            y, sr = sf.read(in_path)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != 48000:
                y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=48000)
                sr = 48000
            y = y.astype(np.float32)
            with torch.no_grad():
                enhanced = enhance(model, df_state, y)
            sf.write(out_path, enhanced, sr)
            return True, f"DeepFilterNet OK (device={device})"
        else:
            raise RuntimeError("Model not loaded.")
    except Exception as e:
        print("DeepFilterNet failed, fallback to spectral gate:", e)
        print(traceback.format_exc())
        try:
            y, sr = librosa.load(in_path, sr=None, mono=True)
            S = librosa.stft(y, n_fft=2048, hop_length=512)
            p = np.abs(S) ** 2
            n_frames = max(1, int(0.3 * sr / 512))
            noise = np.mean(p[:, :n_frames], axis=1, keepdims=True)
            gate_strength = 2.5
            mask = np.maximum(1 - gate_strength * noise / (p + noise), 0)
            S_clean = S * mask
            y_dn = librosa.istft(S_clean, hop_length=512)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            S2 = librosa.stft(y_dn, n_fft=2048, hop_length=512)
            S2[freqs > 8000, :] = 0
            y_dn = librosa.istft(S2, hop_length=512)
            sf.write(out_path, y_dn, sr)
            return True, "Fallback spectral gating used"
        except Exception as e2:
            return False, f"DeepFilterNet error: {e}\nFallback error: {e2}"

def wpe_dereverb(in_path, out_path, taps=10, delay=3, iterations=2):
    x, sr = sf.read(in_path)
    if x.ndim == 1:
        X = np.expand_dims(x, 0)
    else:
        X = x.T
    Y = wpe(X, taps=taps, delay=delay, iterations=iterations)
    sf.write(out_path, Y[0], sr)

def run_ffmpeg_cmd(cmd):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, proc.stderr
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or "No stderr captured.")
    except FileNotFoundError:
        return False, "ffmpeg not found"

def apply_eq_comp_limiter(in_path, out_path):
    limiter_limit_linear = "0.99"
    af_chain = (
        "highpass=f=80,"
        "equalizer=f=80:width_type=o:width=1:g=6,"
        "equalizer=f=160:width_type=o:width=1:g=6,"
        "equalizer=f=300:width_type=o:width=1:g=6,"
        "equalizer=f=600:width_type=o:width=1:g=6,"
        "equalizer=f=1200:width_type=o:width=1:g=6,"
        "equalizer=f=3000:width_type=o:width=1:g=6,"
        "equalizer=f=6000:width_type=o:width=1:g=-6,"
        "equalizer=f=12000:width_type=o:width=1:g=-6,"
        "acompressor=threshold=-18dB:ratio=2.5:attack=6:release=150,"
        f"alimiter=limit={limiter_limit_linear}"
    )
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", in_path,
           "-af", af_chain, "-ar", "48000", "-ac", "1",
           "-c:a", "pcm_s16le", out_path]
    ok, stderr = run_ffmpeg_cmd(cmd)
    if ok:
        return True, stderr
    st.warning("ffmpeg EQ/Comp/Limiter failed. See error below:")
    st.code(stderr[-800:], language="bash")
    return False, stderr

def normalize_lufs(in_path, out_path, target=-16.0):
    y, sr = librosa.load(in_path, sr=None, mono=True)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    y_norm = pyln.normalize.loudness(y, loudness, target)
    sf.write(out_path, y_norm, sr)

# -------------------------------------------------------------
#  Streamlit UI
# -------------------------------------------------------------
st.set_page_config(page_title="🎙️ AI Voice Cleanup (DeepFilterNet)", layout="centered")
st.title("🎙️ AI Voice Cleanup MVP — DeepFilterNet Edition")
st.write("Upload a **raw voice recording**. The app uses DeepFilterNet for denoising, "
         "then dereverb, EQ, compress, limit, and normalize to **-16 LUFS**.")

uploaded_file = st.file_uploader("Upload input audio", type=["wav","mp3","m4a","flac"])
preset = st.selectbox("Preset", ["Podcast-calm", "Dramatic"], index=0)
process_btn = st.button("🚀 Clean & Process")

if uploaded_file is not None:
    y, sr = load_audio(uploaded_file.read())
    st.audio(save_temp_audio(y, sr), format="audio/wav")
    st.info(f"Loaded audio • {sr} Hz • {len(y)/sr:.1f}s")

if process_btn and uploaded_file is not None:
    with st.spinner("Loading DeepFilterNet …"):
        model, df_state, device, err = load_df_model()
        if model is None:
            st.warning(f"DeepFilterNet load failed: {err or 'unknown error'}. Will use spectral fallback.")
        else:
            st.success(f"DeepFilterNet loaded on {device.upper()} ✅")

    with st.spinner("Processing audio … please wait ⏳"):
        tmp_input = save_temp_audio(y, sr)
        tmp_denoise = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        tmp_dereverb = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        tmp_eqcomp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        tmp_norm = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        # 1️⃣ Denoise
        st.text("Step 1 — Denoising (DeepFilterNet / Fallback)")
        ok, msg = denoise_deepfilternet(tmp_input, tmp_denoise, model, df_state, device)
        st.text(f"Denoiser: {msg}")

        # 2️⃣ Dereverb
        st.text("Step 2 — Dereverb (WPE)")
        wpe_dereverb(tmp_denoise, tmp_dereverb)

        # 3️⃣ EQ + Compressor + Limiter
        st.text("Step 3 — EQ + Compressor + Limiter (ffmpeg)")
        ok, _ = apply_eq_comp_limiter(tmp_dereverb, tmp_eqcomp)
        if not ok:
            st.stop()

        # 4️⃣ Loudness Normalization
        st.text("Step 4 — Loudness Normalization (−16 LUFS)")
        normalize_lufs(tmp_eqcomp, tmp_norm, target=-16.0)

        # 5️⃣ Output
        st.success("✅ Voice cleanup complete!")
        st.audio(tmp_norm, format="audio/wav")
        with open(tmp_norm, "rb") as f:
            st.download_button("⬇️ Download Cleaned Audio", f, file_name="cleaned_output.wav", mime="audio/wav")

        # Loudness check
        y_clean, _ = librosa.load(tmp_norm, sr=sr, mono=True)
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(y_clean)
        st.caption(f"Final loudness ≈ {lufs:.2f} LUFS (target −16 LUFS)")

        # Cleanup temps
        for p in [tmp_input, tmp_denoise, tmp_dereverb, tmp_eqcomp]:
            try: os.remove(p)
            except: pass
