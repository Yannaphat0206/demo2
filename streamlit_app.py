import openai
import streamlit as st
import langid
import pandas as pd
from deep_translator import GoogleTranslator
from spellchecker import SpellChecker
from pythainlp.spell import correct as thai_correct
import nltk
from nltk.corpus import wordnet
import requests

# Ensure necessary NLTK data is available
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sidebar for API Key
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password", key="api_key")
if user_api_key:
    openai.api_key = user_api_key

# OpenAI Function to get definition using ChatGPT model
def get_openai_definition(word):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": f"Provide a definition for the word '{word}'"}],
            max_tokens=100,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Function to detect language using OpenAI
def detect_language_openai(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": f"Detect the language of the following text: '{text}'"}],
            max_tokens=50,
            temperature=0.5
        )
        language = response['choices'][0]['message']['content'].strip()
        return language
    except Exception as e:
        return f"Error: {str(e)}"

# Function to correct spelling
def correct_spelling(text, lang):
    spell = SpellChecker(language=lang)
    words = text.split()
    corrected_words = [spell.correction(word) for word in words]
    return " ".join(corrected_words)

# Translation function
def translate_input(user_input, detected_lang):
    translations = {}
    try:
        if detected_lang == "fr":
            translations["English"] = GoogleTranslator(source="fr", target="en").translate(user_input)
            translations["Thai"] = GoogleTranslator(source="fr", target="th").translate(user_input)
        elif detected_lang == "en":
            translations["French"] = GoogleTranslator(source="en", target="fr").translate(user_input)
            translations["Thai"] = GoogleTranslator(source="en", target="th").translate(user_input)
        elif detected_lang == "th":
            translations["English"] = GoogleTranslator(source="th", target="en").translate(user_input)
            translations["French"] = GoogleTranslator(source="th", target="fr").translate(user_input)
        else:
            return {}
    except Exception as e:
        return {"Error": f"Translation Error: {str(e)}"}
    return translations

# Get synonyms for different languages
def get_synonyms_and_definitions(word, lang):
    if lang == 'en':
        synonyms = get_synonyms_nltk_english(word)
    elif lang == 'fr':
        synonyms = get_synonyms_french(word)
    elif lang == 'th':
        synonyms = get_synonyms_thai(word)
    else:
        return "Language not supported"

    definitions = [get_openai_definition(syn) for syn in synonyms]
    pos_and_gender = [get_pos_and_gender(syn, lang) for syn in synonyms]
    
    # Create a DataFrame for synonyms, POS, and Definitions
    data = [{"Synonym": syn, "POS & Gender": pos, "Definition": defn} for syn, pos, defn in zip(synonyms, pos_and_gender, definitions)]
    return pd.DataFrame(data)

# Function to get English synonyms using NLTK
def get_synonyms_nltk_english(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    while len(synonyms) < 3:
        synonyms.add(word)
    return list(synonyms)

# Function to get French synonyms using WordReference API (example)
def get_synonyms_french(word):
    try:
        url = f"https://api.wordreference.com/0.8/your_api_key/json/fren/en/{word}"  # Replace with your actual API key
        response = requests.get(url).json()
        if 'term0' in response:
            return [entry['term'] for entry in response['term0']['entries']]
        else:
            return [word]  # Default to the word itself if no synonyms found
    except Exception as e:
        return [f"Error fetching synonyms for {word}: {str(e)}"]  # Return detailed error

# Function to get Thai synonyms using PyThaiNLP
def get_synonyms_thai(word):
    try:
        synonyms = [thai_correct(word)]  # Placeholder; integrate more sophisticated logic as needed
        if len(synonyms) == 0:
            return [f"No synonyms found for {word}"]  # Fallback
        return synonyms
    except Exception as e:
        return [f"Error fetching synonyms for {word}: {str(e)}"]  # Return detailed error

# Function to determine part of speech and gender for French words
def get_pos_and_gender(word, lang):
    if lang == 'fr':
        # French POS and Gender (simplified; you may integrate actual French tools for POS tagging)
        if word.lower() in ['le', 'la', 'un', 'une']:  # Articles
            return "Article", None
        if word.lower() in ['chat', 'chien']:  # Noun examples (simplified)
            return "Noun", "Masculine" if word == 'chat' else "Feminine"
        if word.lower() in ['belle', 'grand']:  # Adjective examples (simplified)
            return "Adjective", "Feminine" if word == 'belle' else "Masculine"
        return "Unknown", None
    return "Unknown", None  # For other languages, return unknown

# Main app
st.title("Translator with Synonym, POS, Gender, and Definition")
st.markdown("Input your vocabulary, and we'll provide translations, synonyms, part of speech, gender (for French), and definitions.")

# User input
user_input = st.text_area("Enter the word or phrase:")

if user_input.strip():  # Ensure non-empty input
    detected_lang = langid.classify(user_input)[0]
    st.write(f"Detected Language: {detected_lang.upper()}")

    # Correct spelling if necessary
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

    # Translate the input and display the result
    translations = translate_input(user_input, detected_lang)
    if translations:
        st.write("Translations:")
        for lang, translation in translations.items():
            st.write(f"- {lang}: {translation}")

        # Get synonyms, POS, gender, and definitions for each language
        for lang, translation in translations.items():
            st.write(f"{lang} Synonyms, POS, Gender, and Definitions:")
            translated_df = get_synonyms_and_definitions(translation, lang)
            if isinstance(translated_df, pd.DataFrame):
                st.dataframe(translated_df)
            else:
                st.write(translated_df)  # Handle error message (e.g., "Language not supported")
else:
    st.warning("Please enter a valid word or phrase to process.")
