import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import whisper
import os
from langdetect import detect, DetectorFactory

# Set a seed to ensure consistent language detection results
DetectorFactory.seed = 0

def record_audio(duration=5, filename="recorded.wav", fs=16000):
    """
    Record audio from the microphone for a given duration.
    """
    print("Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wavfile.write(filename, fs, recording)
    print(f"Recording saved as {filename}")
    return filename

def transcribe_audio(file_path):
    """
    Transcribe an audio file using OpenAI's Whisper model.
    """
    print("Loading Whisper model...")
    model = whisper.load_model("large")  # Using 'large' for best accuracy
    print("Transcribing audio...")
    result = model.transcribe(file_path)
    return {
        "text": result["text"],
        "whisper_language": result.get("language", "unknown")
    }

def detect_language(text):
    """
    Detect language from a given text using langdetect.
    """
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        print("Error in language detection:", e)
        return "unknown"

if __name__ == "__main__":
    # Ensure ffmpeg is installed on your system
    if os.system("ffmpeg -version") != 0:
        print("ffmpeg is not installed. Please install ffmpeg and try again.")
        exit(1)
        
    # Step 1: Record audio
    audio_file = record_audio(duration=5)
    
    # Step 2: Transcribe the recorded audio
    transcription = transcribe_audio(audio_file)
    transcribed_text = transcription["text"]
    whisper_detected_lang = transcription["whisper_language"]
    
    # Step 3: Additional language detection using langdetect
    detected_lang = detect_language(transcribed_text)
    
    print("\n--- Transcription Results ---")
    print("Whisper Detected Language:", whisper_detected_lang)
    print("Langdetect Detected Language:", detected_lang)
    print("Transcribed Text:", transcribed_text)