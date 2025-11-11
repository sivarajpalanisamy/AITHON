import librosa
import soundfile as sf
import noisereduce as nr
import subprocess
from pydub import AudioSegment
from pydub.effects import normalize, low_pass_filter, high_pass_filter
import os

def safe_print(text):
    """Print with proper encoding handling for Windows console"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for Windows console
        print(text.encode('ascii', 'replace').decode('ascii'))

def ffmpeg_pre_clean(input_path: str, pre_output_path: str):
    """
    Step 1: Use FFmpeg native denoise filters before Python processing.
    Good for removing static hum or broadband room noise.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-af", "afftdn=nf=-25,highpass=f=180,lowpass=f=6200",
        pre_output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return pre_output_path


def hybrid_noise_reduce(input_path: str, output_path: str):
    """
    Step 2: Combine FFmpeg denoise + Noisereduce spectral gating + EQ cleanup.
    Produces much cleaner vocals than raw noisereduce.
    """
    safe_print("🧹 Starting hybrid noise cleaning...")

    tmp_path = "temp_ffmpeg_clean.wav"
    ffmpeg_pre_clean(input_path, tmp_path)

    y, sr = librosa.load(tmp_path, sr=48000)
    reduced = nr.reduce_noise(y=y, sr=sr, prop_decrease=1.0, stationary=False)
    sf.write(output_path, reduced, sr)

    safe_print("✅ Noise heavily reduced and exported to: " + output_path)
    return output_path


def normalize_voice(input_path: str, output_path: str):
    """
    Step 3: Loudness normalization and light EQ for clarity.
    """
    safe_print("🎚️ Normalizing and equalizing audio...")

    voice = AudioSegment.from_file(input_path)
    voice = normalize(voice)
    voice = high_pass_filter(voice, cutoff=180)
    voice = low_pass_filter(voice, cutoff=6500)

    voice.export(output_path, format="wav")
    safe_print("✅ Normalized and balanced: " + output_path)
    return output_path


def clean_and_normalize(input_path: str, output_path: str):
    """
    Full pipeline = FFmpeg prefilter → Noisereduce → Normalize + EQ
    """
    os.makedirs("temp", exist_ok=True)
    temp_noise = "temp/noise_clean.wav"
    final_output = "temp/final_cleaned.wav"

    hybrid_noise_reduce(input_path, temp_noise)
    normalize_voice(temp_noise, final_output)
    return final_output


if __name__ == "__main__":
    in_file = "sample_voice.wav"  # or your .mp3 input
    out_file = "cleaned_output.wav"
    clean_and_normalize(in_file, out_file)
