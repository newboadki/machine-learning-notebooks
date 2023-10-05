# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
# TensorFlow
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, TimeDistributed


class DataSets:

    class Set:
        x = list()
        y = list()
        x_vocab_size = 0
        y_vocab_size = 0

        def __init__(self, x_list, y_list):
            self.x = x_list
            self.y = y_list

    train = Set(list(), list())
    val = Set(list(), list())


def tokenized_padded_sets(df,
                          x_tokenizer,
                          y_tokenizer,
                          max_text_len,
                          max_summary_len,
                          test_size_ratio=0.1):
    # Splitting the training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        np.array(df['text']),
        np.array(df['headlines']),
        test_size=test_size_ratio,
        random_state=1,
        shuffle=True)

    # Tokenization
    x_tokenizer.fit_on_texts(list(x_train))
    x_train_sequence = x_tokenizer.texts_to_sequences(x_train)
    x_val_sequence = x_tokenizer.texts_to_sequences(x_val)

    y_tokenizer.fit_on_texts(list(y_train))
    y_train_sequence = y_tokenizer.texts_to_sequences(y_train)
    y_val_sequence = y_tokenizer.texts_to_sequences(y_val)

    # Padding
    x_train_padded = pad_sequences(x_train_sequence,
                                   maxlen=max_text_len,
                                   padding='post')
    x_val_padded = pad_sequences(x_val_sequence,
                                 maxlen=max_text_len,
                                 padding='post')
    y_train_padded = pad_sequences(y_train_sequence,
                                   maxlen=max_summary_len,
                                   padding='post')
    y_val_padded = pad_sequences(y_val_sequence,
                                 maxlen=max_summary_len,
                                 padding='post')

    # Vocab size
    x_vocab_size = len(x_tokenizer.word_index) + 1
    y_vocab_size = len(y_tokenizer.word_index) + 1

    # Result
    ds = DataSets()
    ds.x_vocab_size = x_vocab_size
    ds.y_vocab_size = y_vocab_size
    ds.train = DataSets.Set(x_train_padded, y_train_padded)
    ds.val = DataSets.Set(x_val_padded, y_val_padded)

    return ds

def get_embedding_matrix(tokenizer, embedding_dim, vocab_size=None):
    word_index = tokenizer.word_index
    voc = list(word_index.keys())

    path_to_glove_file = './data/glove.6B.100d.txt'

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    num_tokens = len(voc) + 2 if not vocab_size else vocab_size
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix