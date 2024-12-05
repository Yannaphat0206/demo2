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

# Preprocessing function to remove articles
def remove_articles(text, lang):
    articles = {
        "fr": ["le", "la", "les", "un", "une", "des", "du", "de", "de la", "de l'", "l'"],
        "en": ["the", "a", "an"],
        "th": []  # Thai does not use articles
    }
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in articles.get(lang, [])]
    return " ".join(filtered_words)

# OpenAI Function to get batch definitions and POS using ChatGPT model
@st.cache_data
def get_openai_definitions_and_pos(words, lang):
    try:
        prompt = f"Provide the part of speech, and definition for each of the following {lang} words:\n" + "\n".join(words)
        if lang == 'French':
            prompt += "\nAlso provide the gender for nouns and adjectives."

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        definitions = response['choices'][0]['message']['content'].strip().split("\n")
        return definitions
    except Exception as e:
        return [f"Error: {str(e)}"]

# Spelling correction function
def correct_spelling(text, lang):
    spell = SpellChecker(language=lang)
    words = text.split()
    corrected_words = [spell.correction(word) for word in words]
    return " ".join(corrected_words)

# Translation function
@st.cache_data
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

# Function to handle synonyms, POS, and definitions
def get_synonyms_pos_and_definitions(word, lang):
    if lang == 'en':
        synonyms = get_synonyms_nltk_english(word)
    elif lang == 'fr':
        synonyms = get_synonyms_french(word)
    elif lang == 'th':
        synonyms = get_synonyms_thai(word)
    else:
        return "Language not supported"
    
    definitions = get_openai_definitions_and_pos(synonyms, lang)
    data = [{"Synonym": syn, "POS/Details": defn} for syn, defn in zip(synonyms, definitions)]
    
    return pd.DataFrame(data)

# Synonyms for English using NLTK
def get_synonyms_nltk_english(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if len(synonyms) >= 5:
                break
            synonyms.add(lemma.name())
    return list(synonyms)

# Placeholder synonyms for French
def get_synonyms_french(word):
    synonyms = [word]  # Placeholder
    while len(synonyms) < 3:
        synonyms.append(word)
    return synonyms

# Placeholder synonyms for Thai
def get_synonyms_thai(word):
    synonyms = [word]
    while len(synonyms) < 3:
        synonyms.append(word)
    return synonyms

# Title for the app
st.title("Translator with Synonyms, POS, and Definitions")
st.markdown("Input your vocabulary, and we'll provide translations, synonyms, POS, and definitions.")

# User input
user_input = st.text_area("Enter the word or phrase:")

if user_input.strip():
    detected_lang = langid.classify(user_input)[0]
    st.write(f"Detected Language: {detected_lang.upper()}")

    # Remove articles
    cleaned_input = remove_articles(user_input, detected_lang)
    st.write(f"Processed Word Without Articles: {cleaned_input}")

    # Spelling correction
    if detected_lang in ["en", "fr"]:
        corrected_word = correct_spelling(cleaned_input, detected_lang)
        if corrected_word != cleaned_input:
            st.write(f"Corrected Spelling: {corrected_word}")
            cleaned_input = corrected_word
    elif detected_lang == "th":
        corrected_word = thai_correct(cleaned_input)
        if corrected_word != cleaned_input:
            st.write(f"Corrected Spelling: {corrected_word}")
            cleaned_input = corrected_word

    # Synonyms, POS, and definitions for the original language
    st.write(f"{detected_lang.upper()} Synonyms, POS, and Definitions:")
    original_df = get_synonyms_pos_and_definitions(cleaned_input, detected_lang)
    st.dataframe(original_df)

    # Translate the input and display the result
    translations = translate_input(cleaned_input, detected_lang)
    if translations:
        st.write("Translations:")
        for lang, translation in translations.items():
            st.write(f"- {lang}: {translation}")
            st.write(f"{lang} Synonyms, POS, and Definitions:")
            translated_df = get_synonyms_pos_and_definitions(translation, lang.capitalize())
            st.dataframe(translated_df)
else:
    st.warning("Please enter a valid word or phrase to process.")
