# -*- coding: utf-8 -*-

import string
import pandas as pd
import pickle
import re
import unicodedata
import numpy as np
from bs4 import BeautifulSoup  # Removing HTML

# NLP
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from contractions import contractions_dict

stop_words = set(stopwords.words('english'))


def expand_contractions(text, contraction_map=contractions_dict):
    # Using regex for getting all contracted words
    contractions_keys = '|'.join(contraction_map.keys())
    contractions_pattern = re.compile(f'({contractions_keys})',
                                      flags=re.DOTALL)

    def expand_match(contraction):
        # Getting entire matched sub-string
        match = contraction.group(0)
        expanded_contraction = contraction_map.get(match)
        if not expand_contractions:
            return match
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text


def remove_possessive_apostrophe(text):
    return re.sub(r"'s\b", "", text)


def remove_punctuation_from_text(text):
    return re.sub("[^a-zA-Z]", " ", text)


def remove_numbers_from_text(text):
    text = re.sub('[0-9]+', '', text)
    return ' '.join(text.split())  # to remove `extra` white space


def remove_stopwords_from_text(text, stopwords):
    word_list = [word for word in text.split() if not word in stopwords]
    return ' '.join(word_list)


def remove_special_characters(text):
    # Remove special hyphens (–). This is different from normal hyphen(-)
    text = re.sub('–', '', text)
    text = ' '.join(text.split())  # removing `extra` white spaces

    # Removing white space characters from text
    text = re.sub("(\\t)", ' ', text)
    text = re.sub("(\\r)", ' ', text)
    text = re.sub("(\\n)", ' ', text)

    # remove accented chars ('Sómě Áccěntěd těxt' => 'Some Accented text')
    text = unicodedata.normalize('NFKD', text).encode('ascii',
                                                      'ignore').decode(
                                                          'utf-8', 'ignore')

    text = re.sub("(__+)", ' ', text)
    text = re.sub("(--+)", ' ', text)
    text = re.sub("(~~+)", ' ', text)
    text = re.sub("(\+\++)", ' ', text)
    text = re.sub("(\.\.+)", ' ', text)
    text = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(text)).lower()
    text = re.sub("(mailto:)", ' ', text)
    text = re.sub(r"(\\x9\d)", ' ', text)
    text = re.sub("([iI][nN][cC]\d+)", 'INC_NUM', text)
    text = re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', text)
    text = re.sub("(\.\s+)", ' ', text)
    text = re.sub("(\-\s+)", ' ', text)
    text = re.sub("(\:\s+)", ' ', text)
    text = re.sub("(\s+.\s+)", ' ', text)
    text = re.sub("'", "", text)

    try:
        url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', text)
        repl_url = url.group(3)
        text = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, text)
    except Exception as e:
        pass

    text = re.sub("(\s+)", ' ', text)
    text = re.sub("(\s+.\s+)", ' ', text)

    return text


def remove_indexes(summary_array):
    remove_indexes = []
    for i in range(len(summary_array)):
        count = 0
        for j in summary_array[i]:
            if j != 0:
                count += 1
        if count == 2:
            remove_indexes.append(i)
    return remove_indexes

def remove_long_text_and_summary_from_data_frame(df, max_text_len, max_summary_len, text_key, summary_key):
    """
    Remove texts and summaries with a length larger than the limit
    """
    cleaned_text = np.array(df[text_key])
    cleaned_summary = np.array(df[summary_key])

    short_text = []
    short_summary = []

    for i in range(len(cleaned_text)):
        if len(cleaned_text[i].split()) <= max_text_len and len(
            cleaned_summary[i].split()
        ) <= max_summary_len:
            short_text.append(cleaned_text[i])
            short_summary.append(cleaned_summary[i])

    df = pd.DataFrame({text_key: short_text, summary_key: short_summary})
    return df

def preprocess_text(text, stemming=False):
    text = str.lower(text)
    text = BeautifulSoup(text, "lxml").text  # Remove html tags
    text = re.sub(r'https?://[^\s\n\r]+', '', text)  # Remove hyperlinks
    text = expand_contractions(text)
    text = remove_possessive_apostrophe(text)
    text = remove_punctuation_from_text(text)
    text = remove_numbers_from_text(text)
    text = remove_stopwords_from_text(text, stop_words)
    text = remove_special_characters(text)

    if not stemming:
        return text

    stemmer = PorterStemmer()
    words = text.split(' ')
    stemmed_words = []
    for word in words:
        if (word not in stop_words and word not in string.punctuation):
            stemmed_words.append(stemmer.stem(word))

    return ' '.join(stemmed_words)


def add_markers_to_summary_labels(df, start_token, end_token):
    df.headlines = df.headlines.apply(lambda x: f'{start_token} {x} {end_token}')

def preprocess_data_frame(df, stemming=False, start_token='tokenstart', end_token='tokenend'):
    df.headlines = df.headlines.apply(
        (lambda x: preprocess_text(x, stemming=stemming)))
    df.text = df.text.apply((lambda x: preprocess_text(x, stemming=stemming)))
    add_markers_to_summary_labels(df, start_token=start_token, end_token=end_token)
