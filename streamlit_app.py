import openai
import streamlit as st
import langid
import pandas as pd
from deep_translator import GoogleTranslator
from spellchecker import SpellChecker
from pythainlp.spell import correct as thai_correct
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sidebar for API Key
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password", key="api_key")
if user_api_key:
    openai.api_key = user_api_key

# OpenAI Function to get definition
def get_openai_definition(word):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Provide a definition for the word '{word}'",
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Spelling correction function
def correct_spelling(text, lang):
    spell = SpellChecker(language=lang)
    words = text.split()  # Split the input into words
    corrected_words = [spell.correction(word) for word in words]
    corrected_text = " ".join(corrected_words)
    return corrected_text


# Title for the app
st.title("Translator with Synonym and Definition")
st.markdown("Input your vocabulary, and we'll provide translations, synonyms, and definitions.")

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

# Function to handle synonyms for different languages
def get_synonyms_and_definitions(word, lang):
    if lang == 'en':
        synonyms = get_synonyms_nltk_english(word)
    elif lang == 'fr':
        synonyms = get_synonyms_french(word)
    elif lang == 'th':
        synonyms = get_synonyms_thai(word)
    else:
        return "Language not supported"
    
    # Fetch definitions using OpenAI
    definitions = [get_openai_definition(syn) for syn in synonyms]
    data = [{"Synonym": syn, "Definition": defn} for syn, defn in zip(synonyms, definitions)]
    
    return pd.DataFrame(data)

# Updated synonym and definition functions (examples, you can customize)
def get_synonyms_nltk_english(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    # Ensure at least 3 synonyms
    while len(synonyms) < 3:
        synonyms.add(word)  # Add the original word as a fallback
    return list(synonyms)

# Synonyms for French
def get_synonyms_french(word):
    synonyms = [word]  # Example, you should replace this with actual French synonyms
    # Ensure at least 3 synonyms
    while len(synonyms) < 3:
        synonyms.append(word)  # Add the original word as a fallback
    return synonyms

# Synonyms for Thai
def get_synonyms_thai(word):
    synonyms = [word]  # Example, you should replace this with actual Thai synonyms
    # Ensure at least 3 synonyms
    while len(synonyms) < 3:
        synonyms.append(word)  # Add the original word as a fallback
    return synonyms

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
        
        # Get synonyms and definitions for each language
        if 'English' in translations:
            st.write("English Synonyms and Definitions:")
            english_df = get_synonyms_and_definitions(translations["English"], 'en')
            st.dataframe(english_df)
        
        if 'French' in translations:
            st.write("French Synonyms and Definitions:")
            french_df = get_synonyms_and_definitions(translations["French"], 'fr')
            st.dataframe(french_df)
        
        if 'Thai' in translations:
            st.write("Thai Synonyms and Definitions:")
            thai_df = get_synonyms_and_definitions(translations["Thai"], 'th')
            st.dataframe(thai_df)
else:
    st.warning("Please enter a valid word or phrase to process.")
