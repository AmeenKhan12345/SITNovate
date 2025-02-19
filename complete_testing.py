import os
import openai
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from langdetect import detect, DetectorFactory
from gtts import gTTS
from playsound import playsound

# Ensure reproducible language detection results
DetectorFactory.seed = 0

# Set your OpenAI API key
openai.api_key = ""  # Replace with your actual API key

def record_audio(duration=5, filename="recorded.wav", fs=16000):
    """
    Record audio from the microphone.
    """
    print("Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wavfile.write(filename, fs, recording)
    print(f"Recording saved as {filename}")
    return filename

def transcribe_audio(file_path, model_size="large"):
    """
    Transcribe audio using OpenAI's Whisper.
    """
    print("Loading Whisper model...")
    model = whisper.load_model(model_size)
    print("Transcribing audio...")
    result = model.transcribe(file_path)
    # result contains keys "text" and "language"
    return result["text"], result.get("language", "unknown")

def detect_language(text):
    """
    Detect the language of the given text using langdetect.
    """
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        print("Error in language detection:", e)
        return "unknown"

def generate_response_gpt(user_query, language):
    """
    Generate a response using GPT-3.5 Turbo via OpenAI's API.
    The system prompt instructs the bot to respond in the detected language.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are Bharat Bhai, a witty, humorous, and knowledgeable tech support assistant. "
                "You provide clear, step-by-step tech advice using local idioms, proverbs, and a friendly, slightly sarcastic tone. "
                "Always include a greeting, detailed instructions, a local proverb, and a friendly closing remark. "
                f"Respond in {language.upper()}."
            )
        },
        {"role": "user", "content": user_query}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=150,
    )
    
    return response.choices[0].message['content']

def text_to_speech(text, lang):
    """
    Convert text to speech using gTTS and play the audio.
    """
    tts = gTTS(text=text, lang=lang)
    filename = "response.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

def process_pipeline():
    # Step 1: Record audio (simulate a spoken query)
    audio_file = record_audio(duration=5)
    
    # Step 2: Transcribe audio using Whisper
    transcribed_text, whisper_lang = transcribe_audio(audio_file, model_size="small")
    print("\n--- Transcribed Text ---")
    print(transcribed_text)
    print("Whisper Detected Language:", whisper_lang)
    
    # Step 3: Detect language using langdetect
    detected_lang = detect_language(transcribed_text)
    print("Langdetect Detected Language:", detected_lang)
    
    # Use detected language if available; otherwise, default to Marathi ("mr")
    language_code = detected_lang if detected_lang != "unknown" else "mr"
    
    # Step 4: Generate a text response using GPT-3.5 Turbo in the detected language
    response_text = generate_response_gpt(transcribed_text, language_code)
    print("\n--- GPT Response ---")
    print(response_text)
    
    # Step 5: Convert the text response to speech and play it
    text_to_speech(response_text, lang=language_code)

if __name__ == "__main__":
    process_pipeline()
