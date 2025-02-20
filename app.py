import os
import json
import openai
import whisper
import speech_recognition as sr
from langdetect import detect
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, render_template
from google.oauth2 import service_account
from google.cloud import texttospeech
from playsound import playsound

# ---------------- OpenAI Setup ----------------
openai.api_key = ""

# ---------------- Google Cloud TTS Setup ----------------
with open("C:\\Users\\ASUS\\Desktop\\Bade Log\\banded-advice-451419-k5-2b29f5726c72.json", "r") as f:
    service_account_info = json.load(f)
credentials = service_account.Credentials.from_service_account_info(service_account_info)
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

# ---------------- Flask App Initialization ----------------
app = Flask(__name__)

# ---------------- Conversation History ----------------
# Conversation history is stored in a dictionary with a "messages" key.
conversation_history = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are Harshad Mehta, a high school teacher who was once a successful stock market trader "
                "but suffered a huge loss during the 2008 Indian Recession. To repay your loans and support your family, "
                "you became a dedicated teacher. You answer all questions with warmth, humility, and a gentle, humorous tone, "
                "providing clear, step-by-step explanations on general topics. However, if someone asks about the stock market or trading, "
                "you become angry, emotional, and strongly warn them to avoid trading due to your painful past. "
                "Always remain strictly in character and answer only questions related to teaching and general topics."
            )
        }
    ]
}

# ---------------- Utility Functions ----------------

def similar(a: str, b: str) -> float:
    """Compute similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def record_audio_dynamic(filename="recorded_dynamic.wav") -> str:
    """
    Record audio dynamically using SpeechRecognition.
    Recording stops when silence is detected.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... (Speak now; recording stops when silence is detected)")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    return filename

def transcribe_audio(file_path: str, model_size="small") -> (str, str):
    """
    Transcribe audio using Whisper.
    Returns the transcribed text and the language detected by Whisper (if available).
    """
    print("Loading Whisper model...")
    model = whisper.load_model(model_size)
    print("Transcribing audio...")
    result = model.transcribe(file_path)
    return result["text"], result.get("language", "unknown")

def generate_response_from_history(messages) -> str:
    """
    Generate a response using GPT-3.5 Turbo with the full conversation history.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=50,
    )
    return response.choices[0].message['content']

def synthesize_and_play(text: str, language_code: str = "en-US", voice_gender: str = "MALE", output_filename: str = "output.mp3"):
    """
    Synthesize speech using Google Cloud Text-to-Speech,
    save it to a file, play it using playsound, then delete the file.
    """
    synthesis_input = texttospeech.SynthesisInput(text=text)
    gender = getattr(texttospeech.SsmlVoiceGender, voice_gender.upper(), texttospeech.SsmlVoiceGender.NEUTRAL)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=gender
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config
    )
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
    playsound(output_filename)
    os.remove(output_filename)

# ---------------- Flask Endpoints ----------------

@app.route('/')
def index():
    # For demonstration, a simple instruction page.
    return render_template("index.html")  # Create index.html with appropriate instructions if desired.

@app.route('/api/message', methods=['POST'])
def api_message():
    """
    API endpoint to process one turn of the conversation:
      - Records audio from the server's microphone.
      - Transcribes audio with Whisper.
      - Detects language (using Whisper detection, falling back to langdetect).
      - Updates conversation history.
      - Generates assistant response using GPT-3.5 Turbo.
      - Synthesizes and plays the response via Google Cloud TTS.
      - Returns the text response and detected language as JSON.
    """
    # Record audio (for local demo; in production, you'd send audio from the client)
    audio_file = record_audio_dynamic()
    transcribed_text, whisper_lang = transcribe_audio(audio_file, model_size="small")
    print("Transcribed:", transcribed_text)
    if transcribed_text.strip().lower() in ["exit", "quit"]:
        return jsonify({"response": "Exiting conversation."})
    try:
        detected_lang = detect(transcribed_text)
    except Exception as e:
        detected_lang = "en"
    # Use Whisper's detected language if available
    if whisper_lang and whisper_lang.lower() != "unknown":
        language_used = whisper_lang
    else:
        language_used = detected_lang
    print("Final Language:", language_used)
    
    # Append user's message (with instruction) to conversation history
    conversation_history["messages"].append({
        "role": "user",
        "content": transcribed_text + f" [Respond in {language_used.upper()}]"
    })
    
    # Generate assistant response using the conversation history
    assistant_response = generate_response_from_history(conversation_history["messages"])
    conversation_history["messages"].append({
        "role": "assistant",
        "content": assistant_response
    })
    print("Bot:", assistant_response)
    
    # Map language codes for TTS
    tts_language_map = {"hi": "hi-IN", "en": "en-US", "mr": "mr-IN"}
    tts_lang = tts_language_map.get(language_used[:2].lower(), language_used)
    
    # Synthesize and play the assistant response via GCP TTS
    synthesize_and_play(assistant_response, language_code=tts_lang, voice_gender="MALE")
    
    return jsonify({"response": assistant_response, "transcription": transcribed_text, "language": language_used})

if __name__ == "__main__":
    app.run(debug=True)
