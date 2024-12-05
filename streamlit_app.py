import openai
import streamlit as st
import langid
import pandas as pd
from deep_translator import GoogleTranslator
from PyDictionary import PyDictionary
import pronouncing
from spellchecker import SpellChecker
from pythainlp.spell import correct as thai_correct
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

# Instantiate the dictionary object
dictionary = PyDictionary()

# Sidebar for API Key
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password", key="api_key")
if user_api_key:
    openai.api_key = user_api_key

# OpenAI Function
def get_openai_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.8,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"
    
# Spelling correction function
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

if user_input.strip():  # Ensure non-empty input
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

    # Translate the input and display the result
    translations = translate_input(user_input, detected_lang)

    if translations:
        st.write("Translations:")
        for lang, translation in translations.items():
            st.write(f"- {lang}: {translation}")
        
        # Get synonyms, IPA, and definitions for each language
        if 'English' in translations:
            st.write("English Synonyms, IPA, and Definitions:")
            english_df = get_synonyms_and_ipa(translations["English"], 'en')
            st.dataframe(english_df)
        
        if 'French' in translations:
            st.write("French Synonyms, IPA, and Definitions:")
            french_df = get_synonyms_and_ipa(translations["French"], 'fr')
            st.dataframe(french_df)
        
        if 'Thai' in translations:
            st.write("Thai Synonyms, IPA, and Definitions:")
            thai_df = get_synonyms_and_ipa(translations["Thai"], 'th')
            st.dataframe(thai_df)
else:
    st.warning("Please enter a valid word or phrase to process.")

# Translation function
def translate_input(user_input, detected_lang):
    # Validate input before attempting translation
    if not user_input or len(user_input.strip()) == 0:
        return "Invalid input: Please enter a valid word or phrase."

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
            return "Language not supported for translation."
    except Exception as e:
        return f"Translation Error: {str(e)}"
    
    return translations

# Synonyms and IPA retrieval functions...
