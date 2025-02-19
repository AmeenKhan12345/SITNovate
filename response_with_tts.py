import openai
from gtts import gTTS
from playsound import playsound
import os

# Set your OpenAI API key
openai.api_key = ""

def generate_response_gpt(user_query):
    messages = [
        {
            "role": "system",
            "content": (
                "You are Bharat Bhai, a witty, humorous, and knowledgeable tech support assistant from Maharashtra. "
                "You provide clear, step-by-step tech advice in Marathi using local idioms, proverbs, and a friendly, slightly sarcastic tone. "
                "Always include a greeting, detailed instructions, a local proverb, and a friendly closing remark."
            )
        },
        {"role": "user", "content": user_query}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=100,
    )
    
    return response.choices[0].message['content']

def text_to_speech(text, lang='mr'):
    tts = gTTS(text=text, lang=lang)
    filename = "response.mp3"
    tts.save(filename)
    playsound(filename)

if __name__ == "__main__":
    user_query = "माझा प्रिंटर काम करत नाही आणि त्याची शाई पुन्हा भरली पाहिजे."
    response_text = generate_response_gpt(user_query)
    
    print("\n--- Bot Text Response ---")
    print(response_text)
    
    # Convert the text response to speech
    text_to_speech(response_text, lang='mr')