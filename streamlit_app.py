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

# Synonyms and IPA for English
def get_synonyms_nltk_english(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def get_ipa_english(word):
    ipa_list = pronouncing.phones_for_word(word)
    if ipa_list:
        return pronouncing.ipa(ipa_list[0])
    return "IPA not found"

# Synonyms and IPA for French
def get_synonyms_french(word):
    return [word]  # Replace with actual API or dictionary call for French synonyms

def get_ipa_french(word):
    return "IPA for French not implemented"  # Replace with actual IPA retrieval

# Synonyms and IPA for Thai
def get_synonyms_thai(word):
    return [word]  # Replace with actual API or dictionary call for Thai synonyms

def get_ipa_thai(word):
    return "IPA for Thai not implemented"  # Replace with actual IPA retrieval

# Function to handle synonyms and IPA for different languages
def get_synonyms_and_ipa(word, lang):
    if lang == 'en':
        synonyms = get_synonyms_nltk_english(word)
        ipa_transcriptions = [get_ipa_english(syn) for syn in synonyms]
    elif lang == 'fr':
        synonyms = get_synonyms_french(word)
        ipa_transcriptions = [get_ipa_french(syn) for syn in synonyms]
    elif lang == 'th':
        synonyms = get_synonyms_thai(word)
        ipa_transcriptions = [get_ipa_thai(syn) for syn in synonyms]
    else:
        return "Language not supported"
    
    # Fetch definitions
    definitions = [get_first_definition(dictionary.meaning(syn)) for syn in synonyms]
    data = [{"Synonym": syn, "IPA": ipa, "Definition": defn} for syn, ipa, defn in zip(synonyms, ipa_transcriptions, definitions)]
    
    return pd.DataFrame(data)

# Definition retrieval
def get_first_definition(defn):
    if defn:
        for pos in defn.values():
            return pos[0]
    return "No definition available"

# Translation function
def translate_input(user_input, detected_lang):
    translations = {}
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
    
    return translations

# Translate the input and display the result
if user_input:
    # Translate the input
    translations = translate_input(user_input, detected_lang)

    if translations:
        st.write(f"Detected Language: {detected_lang.upper()}")
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
