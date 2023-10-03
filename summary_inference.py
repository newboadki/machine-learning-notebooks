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


def decode_sequence(input_seq,
                    max_summary_len,
                    inf_encoder_model,
                    inf_decoder_model,
                    target_word_index,
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


def gru_based_inference_models(model_info, latent_dim, max_text_len):
    model = model_info['model']
    encoder_input = model_info['encoder_input']
    encoder_output = model_info['encoder_output']
    encoder_state = model_info['encoder_state']
    decoder_input = model_info['decoder_input']
    decoder_state = model_info['decoder_state']
    decoder_dense = model_info['decoder_dense']
    y_embedding_layer = model_info['decoder_embedding_layer']
    decoder_gru = model_info['decoder_gru']
    
    # Encoder Inference Model
    encoder_model_inference = Model(encoder_input, [encoder_output, encoder_state])

    # Decoder Inference
    # Below tensors will hold the states of the previous time step
    decoder_state = Input(shape=(latent_dim*2, ))
    decoder_intermittent_state_input = Input(shape=(max_text_len, latent_dim*2))

    # Get Embeddings of Decoder Sequence
    decoder_embedding_inference = y_embedding_layer(decoder_input)

    # Predict Next Word in Sequence, Set Initial State to State from Previous Time Step
    decoder_output_inference, decoder_state_inference = decoder_gru(decoder_embedding_inference,
                                                                    initial_state=[decoder_state])

    # Attention Inference
    attention_layer = AttentionLayer()
    attention_out_inference, attention_state_inference = attention_layer([decoder_intermittent_state_input,
                                                                          decoder_output_inference])
    decoder_inference_concat = Concatenate(axis=-1)([decoder_output_inference,
                                                     attention_out_inference])

    # Dense Softmax Layer to Generate Prob. Dist. Over Target Vocabulary
    decoder_output_inference = decoder_dense(decoder_inference_concat)

    # Final Decoder Model
    decoder_model_inference = Model([decoder_input, decoder_intermittent_state_input, decoder_state], 
                                    [decoder_output_inference, decoder_state_inference])
    
    return encoder_model_inference, decoder_model_inference

def gru_decode_sequence(input_sequence,
                    max_summary_len,
                    enc_inference_model, 
                    dec_inference_model, 
                    start_token, 
                    end_token, 
                    target_word_index,
                    reverse_target_word_index):
  """Text generation function via encoder / decoder network."""

  # Encode Input as State Vectors.
  encoder_output, encoder_state = enc_inference_model.predict(input_sequence)

  # Generate Empty Target Sequence of Length 1.
  target_sequence = np.zeros((1, 1))

  # Choose 'start' as the first word of the target sequence
  target_sequence[0, 0] = target_word_index[start_token]

  decoded_sentence = ''
  break_condition = False
  while not break_condition:
      token_output, decoder_state = dec_inference_model.predict([target_sequence, 
                                                                 encoder_output,
                                                                 encoder_state])

      # Sample Token
      sampled_token_index = np.argmax(token_output[0, -1, :])

      if not sampled_token_index == 0:
        sampled_token = reverse_target_word_index[sampled_token_index]

        if not sampled_token == end_token:
            decoded_sentence += ' ' + sampled_token

        # Break Condition: Encounter Max Length / Find Stop Token.
        if sampled_token == end_token or len(decoded_sentence.split()) >= (max_summary_len - 1):
            break_condition = True

        # Update Target Sequence (length 1).
        target_sequence = np.zeros((1, 1))
        target_sequence[0, 0] = sampled_token_index

      else:
        break_condition = True

      # Update internal states
      encoder_state = decoder_state

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
