import os
from google.oauth2 import service_account
from google.cloud import texttospeech
from playsound import playsound
import json
# Create credentials using the service account info
from google.oauth2 import service_account
from google.cloud import texttospeech

# Load your service account JSON from a file
with open("C:\\Users\\ASUS\\Desktop\\Bade Log\\banded-advice-451419-k5-2b29f5726c72.json", "r") as f:
    service_account_info = json.load(f)

# Create credentials using the service account info
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Create the Text-to-Speech client using these credentials
client = texttospeech.TextToSpeechClient(credentials=credentials)

def synthesize_and_play(text: str, language_code: str = "en-US", voice_gender: str = "MALE", output_filename: str = "output.mp3"):
    """
    Synthesize speech using Google Cloud Text-to-Speech API,
    save it to a file, play the file using playsound, then delete the file.
    """
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Determine the voice gender enum
    gender = getattr(texttospeech.SsmlVoiceGender, voice_gender.upper(), texttospeech.SsmlVoiceGender.NEUTRAL)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=gender
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    
    # Write the audio content to a file
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
    
    # Play the audio file
    from playsound import playsound
    playsound(output_filename)
    
    # Delete the file
    os.remove(output_filename)

if __name__ == "__main__":
    sample_text = "नमस्कार, तुमचा राउटर रीस्टार्ट करा आणि ३० सेकंद थांबा."
    # For example, using a Hindi voice variant (if Marathi is not available)
    synthesize_and_play(sample_text, language_code="hi-IN", voice_gender="MALE", output_filename="output.mp3")
