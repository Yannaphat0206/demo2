import requests
import openai
import streamlit as st
import langid
from deep_translator import GoogleTranslator
from spellchecker import SpellChecker
from pythainlp.spell import correct as thai_correct

# Sidebar input to accept the OpenAI API key
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password", key="api_key")

# Set the OpenAI API key (if provided by user)
if user_api_key:
    openai.api_key = user_api_key

# Function to interact with OpenAI for synonyms and definitions
def get_openai_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.8
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Function to correct spelling for French and English
def correct_spelling(text, lang):
    if lang == "fr":
        spell = SpellChecker(language='fr')
    else:
        spell = SpellChecker(language='en')
    corrected_word = spell.correction(text)
    return corrected_word

# Title for the app
st.title("Translator with Synonym, Definition, and Spelling Correction")
st.markdown("Input your vocabulary, and we'll provide translations, synonyms, definitions, and spelling correction.")

# User input
user_input = st.text_area("Enter the word or phrase:")

if user_input:
    # Detect the language
    detected_lang = langid.classify(user_input)[0]
    st.write(f"Detected Language: {detected_lang.upper()}")

    # Spelling correction
    if detected_lang in ["en", "fr"]:
        corrected_word = correct_spelling(user_input, detected_lang)
        if corrected_word != user_input:
            st.write(f"Corrected Spelling: {corrected_word}")
            user_input = corrected_word
    elif detected_lang == "th":
        corrected_word = thai_correct(user_input)
        if corrected_word != user_input:
            st.write(f"Corrected Spelling: {corrected_word}")
            user_input = corrected_word

    # Translation and Synonym/Definition handling
    if detected_lang == "fr":
        # Translate to English and Thai
        translation_fr_to_en = GoogleTranslator(source='fr', target='en').translate(user_input)
        translation_fr_to_th = GoogleTranslator(source='en', target='th').translate(translation_fr_to_en)
        st.write(f"Translation (French to English): {translation_fr_to_en}")
        st.write(f"Translation (French to Thai): {translation_fr_to_th}")

        # Get OpenAI response
        openai_prompt = f"Provide a synonym and definition for '{translation_fr_to_en}' in English, French, and Thai."
        openai_response = get_openai_response(openai_prompt)
        st.write(f"OpenAI Response: {openai_response}")

    elif detected_lang == "en":
        # Translate to French and Thai
        translation_en_to_fr = GoogleTranslator(source='en', target='fr').translate(user_input)
        translation_en_to_th = GoogleTranslator(source='en', target='th').translate(user_input)
        st.write(f"Translation (English to French): {translation_en_to_fr}")
        st.write(f"Translation (English to Thai): {translation_en_to_th}")

        # Get OpenAI response
        openai_prompt = f"Provide a synonym and definition for '{user_input}' in English, French, and Thai."
        openai_response = get_openai_response(openai_prompt)
        st.write(f"OpenAI Response: {openai_response}")

    elif detected_lang == "th":
        # Translate to English and French
        translation_th_to_en = GoogleTranslator(source='th', target='en').translate(user_input)
        translation_th_to_fr = GoogleTranslator(source='en', target='fr').translate(translation_th_to_en)
        st.write(f"Translation (Thai to English): {translation_th_to_en}")
        st.write(f"Translation (Thai to French): {translation_th_to_fr}")

        # Get OpenAI response
        openai_prompt = f"Provide a synonym and definition for '{translation_th_to_en}' in English, French, and Thai."
        openai_response = get_openai_response(openai_prompt)
        st.write(f"OpenAI Response: {openai_response}")

    else:
        st.write("This language is not supported.")
#