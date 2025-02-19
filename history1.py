import os
import openai
import whisper
import speech_recognition as sr
from langdetect import detect
from difflib import SequenceMatcher
from gtts import gTTS
from playsound import playsound

# Set your OpenAI API key
openai.api_key = ""  # Replace with your actual API key

# Global dictionary to cache responses: {user query: response}
response_cache = {}

def similar(a: str, b: str) -> float:
    """Compute similarity ratio between two strings."""
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

def record_audio_dynamic(filename="recorded_dynamic.wav"):
    """
    Record audio dynamically using SpeechRecognition.
    The recording stops when silence is detected.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... (Speak now; recording stops when silence is detected)")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    return filename

def transcribe_audio(file_path, model_size="small"):
    """
    Transcribe audio using OpenAI's Whisper.
    Returns the transcribed text and the language detected by Whisper (if available).
    """
    print("Loading Whisper model...")
    model = whisper.load_model(model_size)
    print("Transcribing audio...")
    result = model.transcribe(file_path)
    return result["text"], result.get("language", "unknown")

def generate_response_gpt(user_query: str, language: str = "en") -> str:
    """
    Generate a response using GPT-3.5 Turbo via OpenAI's API.
    The system prompt instructs the bot (Bharat Bhai) to respond in the specified language.
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

def text_to_speech(text: str, lang: str = "hi", tld: str = "co.in"):
    """
    Convert text to speech using gTTS and play the audio.
    """
    tts = gTTS(text=text, lang=lang, tld=tld)
    filename = "response.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

def conversation_loop():
    """
    Run a continuous conversation loop that:
      - Dynamically records user audio (using SpeechRecognition)
      - Transcribes the audio with Whisper
      - Detects the language with langdetect
      - Checks for a similar query in the local cache
      - Generates a response via GPT-3.5 Turbo (if no cached response is found)
      - Converts the response to speech using gTTS and plays it
      - Ends the conversation if the transcribed text is "exit" or "quit"
    """
    language_code = None  # Will be determined from the first user input
    print("Start chatting with Bharat Bhai (say 'exit' or 'quit' to end the conversation).")
    
    while True:
        print("\nPlease speak your query:")
        audio_file = record_audio_dynamic()
        transcribed_text, whisper_lang = transcribe_audio(audio_file, model_size="small")
        print("\n--- Transcribed Text ---")
        print(transcribed_text)
        
        # End conversation if the user says "exit" or "quit"
        if transcribed_text.strip().lower() in ["exit", "quit"]:
            print("Exiting conversation.")
            break
        
        # Detect language using langdetect
        try:
            detected_lang = detect(transcribed_text)
            print("Langdetect Detected Language:", detected_lang)
        except Exception as e:
            print("Language detection error:", e)
            detected_lang = "en"
        
        # Set language_code on the first iteration
        if language_code is None:
            language_code = detected_lang
        
        # Check for a cached response
        cached_response = get_cached_response(transcribed_text)
        if cached_response is not None:
            response_text = cached_response
        else:
            response_text = generate_response_gpt(transcribed_text, language=language_code)
            response_cache[transcribed_text] = response_text
        
        print("Bot:", response_text)
        text_to_speech(response_text, lang=language_code)

if __name__ == "__main__":
    conversation_loop()
