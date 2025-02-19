from gtts import gTTS
from playsound import playsound
import os

def text_to_speech(text, lang='mr'):
    """
    Convert text to speech using gTTS and play the audio.
    
    Args:
        text (str): The text to convert to speech.
        lang (str): The language code (default 'mr' for Marathi).
    """
    # Create a gTTS object
    tts = gTTS(text=text, lang=lang)
    filename = "response.mp3"
    
    # Save the audio file
    tts.save(filename)
    
    # Play the audio file
    playsound(filename)
    
    # Optionally, remove the audio file after playing
    os.remove(filename)

# Test the TTS function
if __name__ == "__main__":
    sample_text = "नमस्कार! तुमचा राउटर रीस्टार्ट करा आणि ३० सेकंद थांबा. मग पुन्हा सुरु करा. जर समस्या कायम राहिली तर, तुमच्या सेवा प्रदात्याशी संपर्क साधा."
    text_to_speech(sample_text, lang='mr')