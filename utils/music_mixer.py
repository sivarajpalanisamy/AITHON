from pydub import AudioSegment
import os

def safe_print(text):
    """Print with proper encoding handling for Windows console"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for Windows console
        print(text.encode('ascii', 'replace').decode('ascii'))

def add_music_bed(voice_path: str, music_path: str, output_path: str, volume_db: int = -25):
    """
    Mixes a voice recording with background music using ducking and fade‑in/out.
    The music level is lowered (ducked) to ensure the voice is clear.
    """
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    if not os.path.exists(music_path):
        raise FileNotFoundError(f"Music file not found: {music_path}")

    safe_print("🎵 Loading voice and background music...")
    voice = AudioSegment.from_file(voice_path)
    music = AudioSegment.from_file(music_path)

    # Loop music to match length of voice
    while len(music) < len(voice):
        music += music
    music = music[:len(voice)]

    # Fade‑in and fade‑out music for smoothness
    music = music.fade_in(2000).fade_out(2000)

    # Apply ducking (reduce music volume)
    music = music + volume_db

    safe_print("🎚️ Mixing tracks...")
    mixed = music.overlay(voice)

    mixed.export(output_path, format="mp3")
    safe_print(f"✅ Mixed output saved as: {output_path}")
    return output_path


if __name__ == "__main__":
    voice_test = "cleaned_output.wav"   # output from Module 1
    music_test = "music_beds/calm_acoustic.mp3"
    output_test = "mixed_demo.mp3"
    add_music_bed(voice_test, music_test, output_test)
