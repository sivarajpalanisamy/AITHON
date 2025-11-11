import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize

# Function 1: Basic Noise Reduction
def clean_noise(input_path: str, output_path: str):
    """
    Removes background noise from a raw voice clip using spectral noise reduction.
    """
    print("🔄 Cleaning noise...")
    audio_data, sr = librosa.load(input_path, sr=48000)
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sr)
    sf.write(output_path, reduced_noise, sr)
    print(f"✅ Cleaned audio exported to: {output_path}")
    return output_path


# Function 2: Loudness Normalization
def normalize_audio(input_path: str, output_path: str):
    """
    Adjusts the loudness of the cleaned audio for consistent broadcast-level output.
    """
    print("🎚️ Normalizing audio levels...")
    audio = AudioSegment.from_file(input_path)
    normalized = normalize(audio)
    normalized.export(output_path, format="wav")
    print(f"✅ Normalized file saved to: {output_path}")
    return output_path


# Function 3: Combined Cleanup Pipeline
def clean_and_normalize(input_path: str, output_path: str):
    """
    Combines noise reduction and loudness normalization in one function.
    """
    temp_path = output_path.replace(".wav", "_temp.wav")
    noise_cleaned = clean_noise(input_path, temp_path)
    final_path = normalize_audio(noise_cleaned, output_path)
    return final_path


# Test block (run independently)
if __name__ == "__main__":
    input_test = "sample_voice.wav"  # your raw test file
    output_test = "cleaned_output.wav"
    clean_and_normalize(input_test, output_test)
