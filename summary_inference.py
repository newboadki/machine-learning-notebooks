# == MODEL INFERENCE ==

import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate


def inference_models_from_lstm(model_info, latent_dim, max_text_len):
    """
    Expected keys:
    - model
    - encoder_inputs
    - encoder_outputs
    - state_h
    - state_c
    - dec_emb_layer
    - decoder_inputs
    - decoder_outputs
    - decoder_lstm
    - attn_layer
    - decoder_dense
    """

    model = model_info['model']
    encoder_inputs = model_info['encoder_inputs']
    encoder_outputs = model_info['encoder_outputs']
    state_h = model_info['state_h']
    state_c = model_info['state_c']
    dec_emb_layer = model_info['dec_emb_layer']
    decoder_inputs = model_info['decoder_inputs']
    decoder_outputs = model_info['decoder_outputs']
    decoder_lstm = model_info['decoder_lstm']
    attn_layer = model_info['attn_layer']
    decoder_dense = model_info['decoder_dense']

    # Encoder inference
    encoder_model = Model(inputs=encoder_inputs,
                          outputs=[encoder_outputs, state_h, state_c])

    # Decoder inference
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim, ))
    decoder_state_input_c = Input(shape=(latent_dim, ))
    decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))

    # Get the embeddings of the decoder sequence
    dec_emb2 = dec_emb_layer(decoder_inputs)

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(
        dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    attn_out_inf, attn_states_inf = attn_layer(
        [decoder_hidden_state_input, decoder_outputs2])
    decoder_inf_concat = Concatenate(
        axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_outputs2 = decoder_dense(decoder_inf_concat)

    # Final decoder model
    decoder_model = Model([decoder_inputs] + [
        decoder_hidden_state_input, decoder_state_input_h,
        decoder_state_input_c
    ], [decoder_outputs2] + [state_h2, state_c2])

    return encoder_model, decoder_model


def decode_sequence(input_seq, max_summary_len, inf_encoder_model,
                    inf_decoder_model, target_word_index,
                    reverse_target_word_index):
    # Encode the input as state vectors.
    e_out, e_h, e_c = inf_encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['tokenstart']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = inf_decoder_model.predict([target_seq] +
                                                        [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if (sampled_token != 'tokenend'):
            decoded_sentence += ' ' + sampled_token

            # Exit condition: either hit max length or find stop word.
            if len(decoded_sentence.split()) >= (max_summary_len - 1):
                stop_condition = True
        else:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def seq2summary(input_seq, target_word_index, reverse_target_word_index):
    newString = ''
    for i in input_seq:
        if ((i != 0 and i != target_word_index['tokenstart'])
                and i != target_word_index['tokenend']):
            newString = newString + reverse_target_word_index[i] + ' '
    return newString

def seq2text(input_seq, reverse_source_word_index):
    newString = ''
    for i in input_seq:
        if (i != 0):
            newString = newString + reverse_source_word_index[i] + ' '
    return newString
