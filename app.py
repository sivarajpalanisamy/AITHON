import streamlit as st
import os
import time
from dotenv import load_dotenv
from cleanvoice import Cleanvoice
from pydub import AudioSegment
from utils.music_mixer import add_music_bed
from utils.cue_detection import detect_peaks, add_emphasis_sfx
from utils.intro_outro import add_intro_outro
from openai import OpenAI

# Load environment variables
load_dotenv()
CLEANVOICE_API_KEY = os.getenv("CLEANVOICE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
cv = Cleanvoice({'api_key': CLEANVOICE_API_KEY})

emotion_music_map = {
    "motivation": "music_beds/background_music/bongo-and-drum-instrumental-music-21295.mp3",
    "emotional": "music_beds/background_music/happy-relaxing-loop.mp3",
    "thriller": "music_beds/background_music/horror-background-atmosphere.mp3",
    "suspense": "music_beds/background_music/suspense-piano-theme-131357.mp3",
    "horror": "music_beds/background_music/horror-background-atmosphere.mp3",
    "celebration": "music_beds/background_music/crowd-cheering.mp3",
    "gaming": "music_beds/background_music/game-music-loop.mp3"
}

keyword_sfx_map = {
    "rain": "music_beds/SFX/rain.mp3",
    "thunder": "music_beds/SFX/thunderSound.wav",
    "achievement": "music_beds/SFX/11l-victory_sound_with_t-1749487412779-357604.mp3",
    "birds": "music_beds/SFX/birds-chirping-75156.mp3",
    "train": "music_beds/SFX/indian-train-sound-from-vestibule-realistic-interior-noise-314562.mp3",
    "siren": "music_beds/SFX/tornado-siren-warning-sound-effect.mp3",
    "game": "music_beds/SFX/game-music-loop-1-143979.mp3",
    "ghost": "music_beds/SFX/ghost-horror-sound-382709.mp3"
}

LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "tamil": "ta",
    "hindi": "hi",
    "malayalam": "ml",
    "kannada": "kn",
    "telugu": "te",
    # add other languages here
}

st.set_page_config(page_title="🎙️ Multilingual AI Creator Studio", layout="centered")
st.title("🌐 Multilingual AI Creator Studio")
st.subheader("Upload audio/video; Cleanvoice AI enhances your voice for best results")

# User option checkboxes, default checked
bg_music_opt = st.checkbox("Add Background Music?", value=True)
intro_music_opt = st.checkbox("Add Intro Music?", value=True)
outro_music_opt = st.checkbox("Add Outro Music?", value=True)
sfx_sounds_opt = st.checkbox("Add SFX sounds to enhance?", value=True)

def save_file(uploaded_file):
    dir_path = "Input_audio"
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, uploaded_file.name)
    if os.path.exists(file_path):
        base, ext = os.path.splitext(uploaded_file.name)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        file_path = os.path.join(dir_path, f"{base}_{timestamp}{ext}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def cleanvoice_enhance(local_path, output_path):
    with st.spinner("1️⃣ Enhancing audio with Cleanvoice AI..."):
        result = cv.process(local_path, {
            'fillers': True,
            'long_silences': True,
            'mouth_sounds': True,
            'breath': True,
            'remove_noise': True,
            'normalize': True,
            'sound_studio': True,
            'export_format': 'mp3',
            'target_lufs': -16,
            'transcription': False,
            'summarize': False,
        })
        cv.download_file(result.audio.url, output_path)
        return result, output_path

def whisper_language_detect(audio_path):
    response = client.audio.translations.create(
        file=open(audio_path, "rb"),
        model="whisper-1",
        prompt="Detect language only",
        response_format="verbose_json"
    )
    return response.language

def whisper_transcribe(audio_path, language_code):
    response = client.audio.transcriptions.create(
        file=open(audio_path, "rb"),
        model="whisper-1",
        language=language_code,
        response_format="verbose_json"
    )
    return response.text

def classify_emotion(transcript, language):
    prompt = f"""
You are an expert audio content analyst.

Classify the tone of the following {language} transcription into one of these categories only:
Emotional, Thriller, Comedy, Motivation, Suspense

Transcription:
\"\"\"{transcript}\"\"\"

Respond ONLY with the category name.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Classify {language} transcript emotions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower()

def keyword_positions(text, keywords, audio_len_ms):
    positions = {}
    text_lower = text.lower()
    for keyword in keywords:
        idx = text_lower.find(keyword)
        if idx != -1:
            positions[keyword] = int((idx / len(text_lower)) * audio_len_ms)
    return positions

upload = st.file_uploader("Upload your audio/video", type=["mp3","wav","mp4","mkv","avi"])
process_btn = st.button("✨ Process")

if process_btn and upload:
    file_path = save_file(upload)
    st.write(f"Saved upload: {file_path}")

    enhanced_result, enhanced_path = cleanvoice_enhance(file_path, "temp/enhanced.mp3")
    st.write("✅ Cleanvoice audio enhancement completed.")

    detected_language = whisper_language_detect(enhanced_path)
    st.write(f"Detected language: {detected_language}")

    language_code = LANGUAGE_NAME_TO_CODE.get(detected_language.lower())
    if language_code is None:
        st.warning(f"Unknown language '{detected_language}', defaulting to English")
        language_code = "en"

    transcript = whisper_transcribe(enhanced_path, language_code)
    st.header("Transcription")
    st.text_area("Transcript", transcript, height=250)

    emotion = classify_emotion(transcript, detected_language)
    st.write(f"Detected emotion: {emotion}")

    voice_audio = AudioSegment.from_file(enhanced_path)

    # Background music conditional
    if bg_music_opt:
        music_file = emotion_music_map.get(emotion, list(emotion_music_map.values())[0])
        music_audio = AudioSegment.from_file(music_file) - 20  # Reduce music volume
        repeat_times = (len(voice_audio) // len(music_audio)) + 1
        music_audio = (music_audio * repeat_times)[:len(voice_audio)]
        # Add fade in/out for smooth music
        music_audio = music_audio.fade_in(2000).fade_out(2000)
        # CORRECT: Put music as base, voice on top (so voice is clear)
        mixed_audio = music_audio.overlay(voice_audio)
    else:
        mixed_audio = voice_audio

    # SFX overlay conditional with ducking and no fade-in
    if sfx_sounds_opt:
        keywords_found = {k: v for k,v in keyword_sfx_map.items() if k in transcript.lower()}
        st.write(f"Detected keywords for SFX: {list(keywords_found.keys())}")
        positions = keyword_positions(transcript, keywords_found.keys(), len(mixed_audio))
        for k, pos in positions.items():
            sfx = AudioSegment.from_file(keyword_sfx_map[k]) - 12  # base reduction
            voice_segment = mixed_audio[pos:pos+len(sfx)]
            voice_loudness = voice_segment.dBFS
            if voice_loudness > -20:
                sfx = sfx - 6  # further reduce if voice loud
            sfx = sfx.fade_out(100)  # only fade out
            mixed_audio = mixed_audio.overlay(sfx, position=pos)

    temp_path = "temp/mixed_with_sfx.mp3" if sfx_sounds_opt else "temp/mixed_no_sfx.mp3"
    mixed_audio.export(temp_path, format="mp3")

    # Intro/outro conditional via add_intro_outro helper
    intro_file = "music_beds/SFX/intro.mp3" if intro_music_opt else None
    outro_file = "music_beds/SFX/outro.mp3" if outro_music_opt else None

    final_output = add_intro_outro(temp_path, intro_file, outro_file, "final_voicepod.mp3")

    st.success("✅ Voicepod ready!")
    st.audio("final_voicepod.mp3")

    with open("final_voicepod.mp3", "rb") as f:
        st.download_button("⬇️ Download Voicepod", f, "final_voicepod.mp3")
