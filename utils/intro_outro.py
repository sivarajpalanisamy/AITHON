from pydub import AudioSegment
import os
import sys

def safe_print(text):
    """Print with proper encoding handling for Windows console"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for Windows console
        print(text.encode('ascii', 'replace').decode('ascii'))

def add_intro_outro(audio_path: str, intro_path: str = None, outro_path: str = None, output_path: str = None, fade_ms: int = 800):
    """
    Adds intro and outro MP3 stingers to the main voice clip.
    Smooth fade transitions and unified MP3 workflow for consistent results.
    Supports optional intro/outro (can be None).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Main audio missing: {audio_path}")

    safe_print("🎵 Loading audio file...")

    # Load main audio file
    main = AudioSegment.from_file(audio_path, format="mp3")
    output_audio = main

    # Add intro if provided
    if intro_path and os.path.exists(intro_path):
        safe_print("🎬 Adding intro...")
        intro = AudioSegment.from_file(intro_path, format="mp3")
        output_audio = intro.append(main.fade_in(fade_ms), crossfade=fade_ms)

    # Add outro if provided
    if outro_path and os.path.exists(outro_path):
        safe_print("🎬 Adding outro...")
        outro = AudioSegment.from_file(outro_path, format="mp3")
        output_audio = output_audio.append(outro.fade_out(fade_ms), crossfade=fade_ms)

    # Export as MP3 (final voicepod)
    output_audio.export(output_path, format="mp3", bitrate="192k")
    safe_print(f"✅ Final voicepod created: {output_path}")
    return output_path


if __name__ == "__main__":
    # Update file names as per your project files
    audio_file = "highlighted_voice.mp3"   # main audio
    intro_file = "stingers/intro.mp3"      # short intro
    outro_file = "stingers/outro.mp3"      # short outro
    output_file = "final_voicepod.mp3"     # export file

    add_intro_outro(audio_file, intro_file, outro_file, output_file)
