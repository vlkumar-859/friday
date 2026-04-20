from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import pyttsx3
import tempfile
import os
from pydub import AudioSegment
from llm import get_llm_response

VOLUME_BOOST_DB = 10  # increase this for more volume (e.g. 15, 20)


print("🧠 Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Model ready.")
def _make_engine():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 170)
    engine.setProperty('volume', 1.0)
    return engine
# =========================
# 🎤 Record Audio Function
# =========================
def record_audio(duration=5, sample_rate=16000):
    print("🎤 Speak now...")

    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype='float32')

    sd.wait()
    print("⏹️ Recording finished")

    audio = audio.flatten()

    # Silence check
    if np.max(np.abs(audio)) < 0.01:
        raise ValueError("No audio detected. Please check your microphone.")

    return audio


# =========================
# 🧠 Transcribe Audio
# =========================
def transcribe_audio(audio):
    print("🔍 Transcribing...")

    segments, _ = model.transcribe(
        audio,
        beam_size=5,
        language="en",
        vad_filter=True
    )

    text = " ".join(segment.text for segment in segments).strip()

    if not text:
        raise ValueError("Could not understand audio")

    return text

def speak(text):
    print("🔊 Speaking...")
    engine = _make_engine()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    engine.save_to_file(text, tmp.name)
    engine.runAndWait()

    # Boost volume and play via sounddevice (no file locking issues)
    boosted = AudioSegment.from_wav(tmp.name) + VOLUME_BOOST_DB
    samples = np.array(boosted.get_array_of_samples(), dtype=np.float32)
    samples /= 2 ** (boosted.sample_width * 8 - 1)  # normalize to [-1, 1]
    os.remove(tmp.name)

    sd.play(samples, samplerate=boosted.frame_rate)
    sd.wait()
    
EXIT_PHRASES = {"quit", "exit", "goodbye", "good bye", "bye", "stop", "see you", "that's all"}

def is_exit(text):
    return any(phrase in text.lower() for phrase in EXIT_PHRASES)


# =========================
# 🚀 Main Execution
# =========================
if __name__ == "__main__":
    print("👂 Listening... (say 'goodbye' or 'quit' to stop)\n")

    while True:
        try:
            audio = record_audio(duration=5)
            user_text = transcribe_audio(audio)

            print("\n🗣️ You said:")
            print("👉", user_text)

            if is_exit(user_text):
                speak("Goodbye! Have a great day.")
                print("👋 Exiting.")
                break

            response = get_llm_response(user_text)

            print("\n🤖 Assistant:")
            print("👉", response)

            speak(response)
            print()

        except ValueError as e:
            # silence or transcription issues — just keep listening
            print(f"⚠️  {e}, listening again...\n")
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Exiting.")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")