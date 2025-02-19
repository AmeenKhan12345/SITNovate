import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import whisper
import os

def record_audio(duration=5, filename="recorded.wav", fs=16000):
    """
    Record audio from the microphone for a given duration.
    
    Args:
        duration (int): Duration to record (in seconds).
        filename (str): The file name to save the recording.
        fs (int): Sampling rate.
        
    Returns:
        str: The path to the saved audio file.
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
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        dict: Contains transcription text and detected language.
    """
    # Load the latest Whisper model (you can choose model size: tiny, base, small, medium, large)
    print("Loading Whisper model...")
    model = whisper.load_model("large", device="cuda")  # "base" is a good balance between speed and accuracy
    
    print("Transcribing audio...")
    result = model.transcribe(file_path)
    
    return {
        "text": result["text"],
        "language": result.get("language", "unknown")
    }

if __name__ == "__main__":
    # Ensure ffmpeg is installed in your system (required by Whisper)
    if os.system("ffmpeg -version") != 0:
        print("ffmpeg is not installed. Please install ffmpeg and try again.")
        exit(1)
        
    # Step 1: Record audio
    audio_file = record_audio(duration=5)  # Adjust duration as needed

    # Step 2: Transcribe the recorded audio
    transcription = transcribe_audio(audio_file)
    
    print("\n--- Transcription Results ---")
    print("Detected Language:", transcription["language"])
    print("Transcribed Text:", transcription["text"])