import streamlit as st
import openai
import langid
from spellchecker import SpellChecker
from pythainlp.spell import correct as thai_correct
import pandas as pd
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

# Sidebar for API Key
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password", key="api_key")
if user_api_key:
    openai.api_key = user_api_key

# Title for the app
st.title("Multilingual Word Analyzer")
st.markdown(
    "Enter a word or phrase in Thai, French, or English to get its definition, translation, and synonyms in different languages."
)

# Helper Functions
def detect_language(text):
    lang, _ = langid.classify(text)
    supported_languages = {"en": "English", "fr": "French", "th": "Thai"}
    return supported_languages.get(lang, f"{lang} not supported")

def correct_spelling(text, lang):
    if lang == "Thai":
        corrected = thai_correct(text)
    else:
        spell = SpellChecker(language="fr" if lang == "French" else "en")
        corrected = " ".join([spell.correction(word) for word in text.split()])
    return corrected if corrected != text else None

def get_openai_translation(word, source_lang, target_lang):
    prompt = f"Translate the word '{word}' from {source_lang} to {target_lang}."
    return get_openai_response(prompt)

def get_openai_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def fetch_definition(word, lang):
    prompt = f"Provide the definition of the word '{word}' in {lang}."
    return get_openai_response(prompt)

def fetch_synonyms(word, lang):
    prompt = (
        f"List at least 4 synonyms for the word '{word}' in {lang}. "
        f"Provide definitions for each synonym as a table with two columns: Synonym and Definition."
    )
    response = get_openai_response(prompt)

    # Attempt to parse response into a DataFrame
    try:
        data = []
        for line in response.split("\n"):
            if ":" in line:
                synonym, definition = line.split(":", 1)
                data.append({"Synonym": synonym.strip(), "Definition": definition.strip()})
        if data:
            return pd.DataFrame(data)
    except Exception:
        pass
    return response or "No synonyms found."

# Main Application Logic
user_input = st.text_input("Enter a word or phrase:")

if user_input.strip() and user_api_key:
    # Step 1: Detect language
    detected_language = detect_language(user_input)
    st.write(f"Detected Language: {detected_language}")

    if "not supported" in detected_language:
        st.error(f"Language not supported: {detected_language}")
    else:
        # Step 2: Correct spelling
        corrected_word = correct_spelling(user_input, detected_language)
        if corrected_word:
            st.warning(f"This isn't correct spelling. Did you mean: {corrected_word}?")
            user_input = corrected_word

        # Step 3: Fetch translations
        translations = {}
        for target_lang in ["English", "French", "Thai"]:
            if target_lang != detected_language:
                translations[target_lang] = get_openai_translation(user_input, detected_language, target_lang)
        
        st.write("Translations:")
        for lang, translation in translations.items():
            st.write(f"- {lang}: {translation}")

        # Step 4: Fetch definitions
        st.write("Definitions:")
        definition_original = fetch_definition(user_input, detected_language)
        st.write(f"In {detected_language}: {definition_original}")
        for lang, translation in translations.items():
            definition_translation = fetch_definition(translation, lang)
            st.write(f"In {lang}: {definition_translation}")

        # Step 5: Fetch synonyms in the same language
        st.subheader(f"Synonyms in {detected_language} (Original Language):")
        synonyms_original = fetch_synonyms(user_input, detected_language)
        if isinstance(synonyms_original, pd.DataFrame):
            st.table(synonyms_original)

        # Step 6: Fetch synonyms in other languages
        st.write("Synonyms in Other Languages:")
        for lang, translation in translations.items():
            st.subheader(f"Synonyms in {lang}:")
            synonyms_data = fetch_synonyms(translation, lang)
            if isinstance(synonyms_data, pd.DataFrame):
                st.table(synonyms_data)
else:
    st.warning("Please provide valid input and API Key.")
