import requests
import openai
import streamlit as st
import langid
import pandas as pd
from deep_translator import GoogleTranslator
from PyDictionary import PyDictionary
import pronouncing

# Initialize Dictionary and Translator
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

# IPA Fetching Function
def get_ipa(word):
    ipa_list = pronouncing.phones_for_word(word)
    if ipa_list:
        return pronouncing.ipa(ipa_list[0])
    else:
        return "IPA not found"

# Synonym and IPA Retrieval Function
def get_synonyms_and_ipa(word, lang="en"):
    synonyms = dictionary.synonym(word) or []
    ipa_transcriptions = [get_ipa(syn) for syn in synonyms[:3]]
    data = [{"Synonym": syn, "IPA": ipa} for syn, ipa in zip(synonyms[:3], ipa_transcriptions)]
    return pd.DataFrame(data)

# App Title
st.title("Translator and Synonym Finder with IPA")
st.markdown("Input your word, and we'll provide translations, synonyms, and their IPA.")

# User Input
user_input = st.text_area("Enter the word or phrase:")

# Detect Language
check_lang = langid.classify(user_input)

if user_input:
    # Handle French Input
    if check_lang[0] == "fr":
        translation_fr_to_en = GoogleTranslator(source='fr', target='en').translate(user_input)
        st.write(f"Translation (French to English): {translation_fr_to_en}")
        
        df_synonyms_ipa = get_synonyms_and_ipa(translation_fr_to_en)
        st.write("Synonyms and IPA Transcription (French -> English):")
        st.dataframe(df_synonyms_ipa)  # Display the DataFrame as a table

    # Handle English Input
    elif check_lang[0] == "en":
        df_synonyms_ipa = get_synonyms_and_ipa(user_input)
        st.write("Synonyms and IPA Transcription (English):")
        st.dataframe(df_synonyms_ipa)  # Display the DataFrame as a table

    # Handle Thai Input
    elif check_lang[0] == "th":
        translation_th_to_en = GoogleTranslator(source='th', target='en').translate(user_input)
        st.write(f"Translation (Thai to English): {translation_th_to_en}")
        
        df_synonyms_ipa = get_synonyms_and_ipa(translation_th_to_en)
        st.write("Synonyms and IPA Transcription (Thai -> English):")
        st.dataframe(df_synonyms_ipa)  # Display the DataFrame as a table
    
    else:
        st.write("Language not supported for synonym and IPA feature.")
