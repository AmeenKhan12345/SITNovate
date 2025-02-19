import os
import openai
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from langdetect import detect
from difflib import SequenceMatcher
from gtts import gTTS
from playsound import playsound

# Set your OpenAI API key
openai.api_key = ""  # Replace with your actual API key

# Global dictionary to cache responses: {user query: response}
response_cache = {}

def record_audio(duration=5, filename="recorded.wav", fs=16000):
    """
    Record audio from the microphone for a given duration and save to a file.
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
    Returns the transcribed text and the detected language (if available).
    """
    print("Loading Whisper model...")
    model = whisper.load_model(model_size)
    print("Transcribing audio...")
    result = model.transcribe(file_path)
    return result["text"], result.get("language", "unknown")

def similar(a: str, b: str) -> float:
    """
    Compute a similarity ratio between two strings.
    """
    return SequenceMatcher(None, a, b).ratio()

def get_cached_response(user_query: str, threshold: float = 0.8) -> str:
    """
    Check if a similar query exists in the cache.
    If a cached query has a similarity ratio above the threshold, return its response.
    """
    for cached_query, cached_response in response_cache.items():
        if similar(user_query.lower(), cached_query.lower()) >= threshold:
            print(f"Found cached response (similarity: {similar(user_query, cached_query):.2f}).")
            return cached_response
    return None

def generate_response_gpt(user_query: str, language: str = "en") -> str:
    """
    Generate a response using GPT-3.5 Turbo via the OpenAI API.
    The system prompt instructs the assistant (Bharat Bhai) to respond in the specified language.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are Bharat Bhai, a witty, humorous, and knowledgeable tech support assistant. "
                "You provide clear, step-by-step technical advice using local idioms and a friendly, slightly sarcastic tone. "
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
        max_tokens=50,
    )
    return response.choices[0].message['content']

def text_to_speech(text: str, lang: str = "en"):
    """
    Convert text to speech using gTTS and play the audio.
    """
    tts = gTTS(text=text, lang=lang)
    filename = "response.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

def conversation_loop():
    """
    Run a continuous conversation loop that:
      1. Records user audio.
      2. Transcribes it using Whisper.
      3. Detects the language.
      4. Checks for a cached response.
      5. Generates a response via GPT-3.5 Turbo if needed.
      6. Converts the response to speech and plays it.
    """
    language_code = None  # To be determined from the first user input
    print("Start chatting with Bharat Bhai (say 'exit' to quit).")
    
    while True:
        print("\nPlease speak your query (recording for 5 seconds)...")
        audio_file = record_audio(duration=5)
        transcribed_text, whisper_lang = transcribe_audio(audio_file, model_size="small")
        print("\n--- Transcribed Text ---")
        print(transcribed_text)
        print("Whisper Detected Language:", whisper_lang)
        
        try:
            detected_lang = detect(transcribed_text)
            print("Langdetect Detected Language:", detected_lang)
        except Exception as e:
            print("Language detection error:", e)
            detected_lang = "en"
        
        # Use detected language if available; otherwise, default to English
        if language_code is None:
            language_code = detected_lang
        
        # Check if user said "exit" or "quit" in their transcription
        if transcribed_text.strip().lower() in ["exit", "quit"]:
            break
        
        # Check the cache for a similar query
        cached_response = get_cached_response(transcribed_text)
        if cached_response is not None:
            response_text = cached_response
        else:
            response_text = generate_response_gpt(transcribed_text, language=language_code)
            # Cache the new query and its response
            response_cache[transcribed_text] = response_text
        
        print("Bot:", response_text)
        text_to_speech(response_text, lang=language_code)

if __name__ == "__main__":
    conversation_loop()
