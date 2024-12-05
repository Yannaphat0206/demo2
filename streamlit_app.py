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

# IPA Fetching Function
def get_ipa(word):
    ipa_list = pronouncing.phones_for_word(word)
    if ipa_list:
        return pronouncing.ipa(ipa_list[0])
    return "IPA not found"

# Synonyms, IPA, and Definitions Retrieval
def get_synonyms_and_ipa(word):
    synonyms = get_synonyms_nltk(word)
    ipa_transcriptions = [get_ipa(syn) for syn in synonyms]
    definitions = [get_first_definition(dictionary.meaning(syn)) for syn in synonyms]
    data = [{"Synonym": syn, "IPA": ipa, "Definition": defn} for syn, ipa, defn in zip(synonyms, ipa_transcriptions, definitions)]
    return pd.DataFrame(data)

# Synonyms using NLTK
def get_synonyms_nltk(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Extract the first definition from dictionary output
def get_first_definition(defn):
    if defn:
        for pos in defn.values():
            return pos[0]
    return "No definition available"

# Spelling Correction Function
def correct_spelling(text, lang):
    if lang == "fr":
        spell = SpellChecker(language='fr')
    elif lang == "en":
        spell = SpellChecker(language='en')
    else:
        return thai_correct(text) if lang == "th" else text
    return spell.correction(text)

# App Title
st.title("Multilingual Translator with Synonyms, IPA, and Definitions")
st.markdown("Input your word or phrase for translation, synonym lookup, IPA transcription, and definition.")

# User Input
user_input = st.text_area("Enter the word or phrase:")

if user_input:
    # Detect the language
    detected_lang = langid.classify(user_input)[0]
    st.write(f"Detected Language: {detected_lang.upper()}")

    # Correct spelling based on detected language
    corrected_word = correct_spelling(user_input, detected_lang)
    if corrected_word != user_input:
        st.write(f"Corrected Spelling: {corrected_word}")
        user_input = corrected_word

    # Translation and Synonym handling based on detected language
    translations = {}
    if detected_lang == "fr":
        translations["English"] = GoogleTranslator(source="fr", target="en").translate(user_input)
        translations["Thai"] = GoogleTranslator(source="en", target="th").translate(translations["English"])
    elif detected_lang == "en":
        translations["French"] = GoogleTranslator(source="en", target="fr").translate(user_input)
        translations["Thai"] = GoogleTranslator(source="en", target="th").translate(user_input)
    elif detected_lang == "th":
        translations["English"] = GoogleTranslator(source="th", target="en").translate(user_input)
        translations["French"] = GoogleTranslator(source="en", target="fr").translate(translations["English"])
    else:
        st.write("Language not supported for translation.")
    
    # Display Translations
    st.write("**Translations:**")
    for lang, translation in translations.items():
        st.write(f"- {lang}: {translation}")

    # Generate Synonyms Table
    synonyms_df = get_synonyms_and_ipa(translations.get("English", user_input))
    st.write("**Synonyms, IPA, and Definitions:**")
    st.dataframe(synonyms_df)
