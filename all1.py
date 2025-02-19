import os
import json
import openai
import whisper
import speech_recognition as sr
from langdetect import detect
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

# ---------------- Conversation History ----------------
# We set a static system prompt that instructs the assistant to respond only in the user's language.
conversation_history = {
    "messages": [
        {
            "role": "system",
            "content": (
                "You are Harshad Mehta, a high school teacher with a unique past. "
                "Once, you were a successful stock market trader, but you suffered a catastrophic loss during the 2008 Indian Recession. That loss forced you to take on massive loans and support your family, so you reinvented yourself as a dedicated teacher. "
                "You now answer all questions with warmth, humility, and a gentle, humorous tone, always offering clear, step-by-step explanations on general topics. "
                "However, if someone asks you about the stock market or trading, you become visibly angry, emotional, and strongly reprimand them—warning them to stay away from trading because of your painful experiences. "
                "Always maintain your kind and humorous approach unless the subject touches on stock market matters, where your past trauma surfaces. Your responses must remain strictly in character, and you should only answer questions related to your new role as a teacher, avoiding any promotion of stock trading."
            )
        }
    ]
}


# ---------------- Utility Functions ----------------

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

def transcribe_audio(file_path: str, model_size="large") -> (str, str):
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
    save it to a file, play it, then delete the file.
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

def conversation_loop():
    """
    Run an audio-only conversation loop:
      - Record user audio.
      - Transcribe using Whisper.
      - Use Whisper's detected language (or fallback to langdetect).
      - Append the user’s message (with an instruction note) to the conversation history.
      - If history exceeds a set length, clear it (or summarize) to avoid context dilution.
      - Generate a response using GPT-3.5 Turbo with the conversation history.
      - Append the assistant’s response to history.
      - Synthesize and play the response via Google Cloud TTS.
      - End if user says "exit" or "quit".
    """
    # We'll use the detected language from Whisper (or fallback) for the conversation.
    language_code = None
    print("Start chatting with Bharat Bhai (say 'exit' or 'quit' to end the conversation).")
    
    turn_count = 0
    while True:
        print("\nPlease speak your query:")
        audio_file = record_audio_dynamic()
        transcribed_text, whisper_lang = transcribe_audio(audio_file, model_size="small")
        print("\n--- Transcribed Text ---")
        print(transcribed_text)
        print("Whisper Detected Language:", whisper_lang)
        
        if transcribed_text.strip().lower() in ["exit", "quit"]:
            print("Exiting conversation.")
            break
        
        # Determine language from Whisper; if unavailable, fallback to langdetect.
        if whisper_lang and whisper_lang.lower() != "unknown":
            language_code = whisper_lang
        else:
            try:
                language_code = detect(transcribed_text)
            except Exception as e:
                print("Language detection error:", e)
                language_code = "en"
        print("Final Language Code:", language_code)
        
        # Append user message to conversation history.
        # Include an instruction note for clarity.
        conversation_history["messages"].append({
            "role": "user",
            "content": transcribed_text + f" [Respond in {language_code.upper()}]"
        })
        
        # Optionally clear history after several turns to maintain context relevance.
        turn_count += 1
        if turn_count >= 6:
            # Clear conversation history except system prompt.
            conversation_history["messages"] = conversation_history["messages"][:1]
            turn_count = 0
        
        # Generate assistant response.
        assistant_response = generate_response_from_history(conversation_history["messages"])
        conversation_history["messages"].append({
            "role": "assistant",
            "content": assistant_response
        })
        print("Bot:", assistant_response)
        
        # Map language codes for TTS.
        tts_language_map = {
            "en": "en-US",
            "hi": "hi-IN",
            "mr": "mr-IN"
        }
        tts_lang = tts_language_map.get(language_code[:2].lower(), language_code)
        
        synthesize_and_play(assistant_response, language_code=tts_lang, voice_gender="MALE")

if __name__ == "__main__":
    conversation_loop()
