import openai

# Set your API key (you can also set this as an environment variable)
openai.api_key = ""  # Replace with your actual OpenAI API key

def generate_response(user_query, personality_info, language):
    """
    Generate a response using GPT-4 based on the user's query, bot's personality, and desired language.
    
    Args:
        user_query (str): The query text from the user.
        personality_info (str): A description of your bot's personality and backstory.
        language (str): The language code in which the response should be generated.
        
    Returns:
        str: The generated response text.
    """
    # Construct the system prompt that establishes the bot's character
    system_prompt = (
        f"{personality_info}\n"
        f"Respond in the language specified: {language}.\n"
        "Always stick to your character and backstory in your answers."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    # Call the GPT-4 API
    try:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=150
)
        generated_response = response.choices[0].message['content'].strip()
        return generated_response
    except Exception as e:
        print("Error generating response:", e)
        return "I'm having trouble processing that request right now."

# Example usage:
if __name__ == "__main__":
    # Example bot personality and backstory
    personality_info = (
        "You are Raj, a witty and humorous Marathi tech support assistant "
        "with a penchant for incorporating local Marathi humor and cultural references. "
        "Always answer with a friendly yet sarcastic tone, and stick strictly to this persona."
    )
    
    # Simulate a user query (this would normally come from your STT & language detection pipeline)
    user_query = "माझ्या संगणकाला इंटरनेट कनेक्शन नाहीये, कृपया मदत करा."
    language = "mr"  # Marathi language code
    
    # Generate and print the response
    bot_response = generate_response(user_query, personality_info, language)
    print("\n--- Bot Response ---")
    print(bot_response)
