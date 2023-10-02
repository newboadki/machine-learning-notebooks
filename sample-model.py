def build_seq2seq_model_with_just_lstm(embedding_dim, latent_dim, max_text_len,
                                       x_vocab_size, y_vocab_size):
    with tpu_strategy.scope():
        K.clear_session()
        latent_dim = 500

        # Encoder
        encoder_inputs = Input(shape=(max_text_len, ))
        enc_emb = Embedding(x_vocab_size, latent_dim,
                            trainable=True)(encoder_inputs)

        #LSTM 1
        encoder_lstm1 = LSTM(latent_dim,
                             return_sequences=True,
                             return_state=True)
        encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

        #LSTM 2
        encoder_lstm2 = LSTM(latent_dim,
                             return_sequences=True,
                             return_state=True)
        encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

        #LSTM 3
        encoder_lstm3 = LSTM(latent_dim,
                             return_state=True,
                             return_sequences=True)
        encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

        # Set up the decoder.
        decoder_inputs = Input(shape=(None, ))
        dec_emb_layer = Embedding(y_vocab_size, latent_dim, trainable=True)
        dec_emb = dec_emb_layer(decoder_inputs)

        # LSTM using encoder_states as initial state
        decoder_lstm = LSTM(latent_dim,
                            return_sequences=True,
                            return_state=True)
        decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(
            dec_emb, initial_state=[state_h, state_c])

        # Attention Layer
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        # Concat attention output and decoder LSTM output
        decoder_concat_input = Concatenate(
            axis=-1, name='concat_layer')([decoder_outputs, attn_out])

        # Dense layer
        decoder_dense = TimeDistributed(
            Dense(y_vocab_size, activation='softmax'))
        decoder_outputs = decoder_dense(decoder_concat_input)

        # Define the model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return {
            'model': model,
            'inputs': {
                'encoder': encoder_inputs,
                'decoder': decoder_inputs
            },
            'outputs': {
                'encoder': encoder_outputs,
                'decoder': decoder_outputs
            },
            'states': {
                'encoder': [state_h, state_c],
                'decoder': [decoder_fwd_state, decoder_back_state]
            },
            'layers': {
                'decoder': {
                    'embedding': dec_emb_layer,
                    'last_decoder_lstm': decoder_lstm,
                    'dense': decoder_dense
                }
            }
        }
