import requests
import openai
import streamlit as st
import langid
from deep_translator import GoogleTranslator  # Import from deep-translator

# Sidebar input to accept the OpenAI API key
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password", key="api_key")

# Set the OpenAI API key (if provided by user)
if user_api_key:
    openai.api_key = user_api_key  # Store the API key to use in requests

# Function to interact with OpenAI (updated for new model)
def get_openai_response(prompt):
    try:
        # Use the OpenAI API to get a response
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Updated model name (replace text-davinci-003)
            prompt=prompt,
            max_tokens=100,  # Limit response length
            n=1,  # Number of responses
            stop=None,  # Stop sequence
            temperature=0.7  # Creativity of response
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"


# Title for the app
st.title("Translator and OpenAI Integration")
st.markdown("Input your vocabulary here, and we'll provide translations, synonyms, and use OpenAI to process the text.")

# Input text from user
user_input = st.text_area("Enter the word or phrase:")

# Detect the language of the input text
check_lang = langid.classify(user_input)

# Example: If the language is French, translate and get OpenAI response for synonym/definition
if check_lang[0] == "fr":
    # Translate French to English using deep-translator
    translation_fr_to_eng = GoogleTranslator(source='fr', target='en').translate(user_input)
    st.write(f"Translation (French to English): {translation_fr_to_eng}")

    # Get a definition or a synonym from OpenAI
    openai_prompt = f"Provide a definition or synonym for the word '{translation_fr_to_eng}' in English."
    openai_response = get_openai_response(openai_prompt)
    st.write(f"OpenAI Response (Definition/Synonym): {openai_response}")

# Example: If the language is English, get a definition or synonym from OpenAI
elif check_lang[0] == "en":
    # Use OpenAI to get a definition or synonym directly for English input
    openai_prompt = f"Provide a definition or synonym for the word '{user_input}' in English."
    openai_response = get_openai_response(openai_prompt)
    st.write(f"OpenAI Response (Definition/Synonym): {openai_response}")

# Example: If the language is Thai, translate to English, then ask OpenAI for a definition or synonym
elif check_lang[0] == "th":
    translation_th_to_en = GoogleTranslator(source='th', target='en').translate(user_input)
    st.write(f"Translation (Thai to English): {translation_th_to_en}")
    
    # Ask OpenAI to define or give synonyms for the translated word
    openai_prompt = f"Provide a definition or synonym for the word '{translation_th_to_en}' in English."
    openai_response = get_openai_response(openai_prompt)
    st.write(f"OpenAI Response (Definition/Synonym): {openai_response}")
else:
    st.write("Language not supported for this feature.")
