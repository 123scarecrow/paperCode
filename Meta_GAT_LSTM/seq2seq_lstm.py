"""
# Data download:
English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip
Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/
# References:
- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import os
import argparse
import random
import io
from pathlib import Path


datalength = 145437

def vectorize_data(data, sample_indices):
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    # lines = codecs.open(data_path, encoding='utf-8').read().split('\n')
    # for line in lines[: min(num_samples, len(lines) - 1)]:
    # lines = data[: min(num_samples, len(data) - 1)]
    lines = [data[i] for i in sample_indices]
    for line in lines:
        input_text, target_text = line.split('\t')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    return input_texts, target_texts, input_characters, target_characters

def prepare_data(data_path, sample_indices, test_indices, shuffle=True):

    if shuffle:
        # Shuffle lines in data and save
        out_data_path = data_path.split('.txt')[0] + '_shuffled.txt'
        with io.open(data_path, 'r', encoding='utf-8') as f:
            data = [(random.random(), line.splitlines()[0]) for line in f]
            data.sort()
            data = [line[1] for line in data]
        with io.open(out_data_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(data))
    else:
        # Shuffle lines in data
        with io.open(data_path, 'r', encoding='utf-8') as f:
            data = [line.splitlines()[0] for line in f]

    # Vectorize the data.
    input_texts, target_texts, input_characters, target_characters = vectorize_data(data, sample_indices)
    input_texts_test, target_texts_test, input_characters_test, target_characters_test = vectorize_data(data, test_indices)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    input_characters_test = sorted(list(input_characters_test))
    target_characters_test = sorted(list(target_characters_test))
    all_input_characters = sorted(list(set(input_characters + input_characters_test)))
    all_target_characters = sorted(list(set(target_characters + target_characters_test)))
    all_input_texts = input_texts + input_texts_test
    all_target_texts = target_texts + target_texts_test
    num_encoder_tokens = len(all_input_characters)
    num_decoder_tokens = len(all_target_characters)
    max_encoder_seq_length = max([len(txt) for txt in all_input_texts])
    max_decoder_seq_length = max([len(txt) for txt in all_target_texts])

    print('Data path:', data_path)
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(all_input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(all_target_characters)])

    # Train data
    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    # Test data
    encoder_test_data = np.zeros(
        (len(input_texts_test), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_test_data = np.zeros(
        (len(input_texts_test), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_test_target_data = np.zeros(
        (len(input_texts_test), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    for i, (input_text_test, target_text_test) in enumerate(zip(input_texts_test, target_texts_test)):
        for t, char in enumerate(input_text_test):
            encoder_test_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text_test):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_test_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_test_target_data[i, t - 1, target_token_index[char]] = 1.

    return input_texts, target_texts, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length,\
        max_decoder_seq_length, input_token_index, target_token_index, encoder_input_data,\
        decoder_input_data, decoder_target_data, reverse_input_char_index, reverse_target_char_index, \
        input_texts_test, target_texts_test, encoder_test_data, decoder_test_data, decoder_test_target_data


# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # return all models
    return model, encoder_model, decoder_model


def load_defined_models(n_units):
    # Save all models
    model = load_model('models/seq2seq_model_{}units.h5'.format(n_units))
    encoder_model = load_model('models/seq2seq_encoder_model_{}units.h5'.format(n_units))
    decoder_model = load_model('models/seq2seq_decoder_model_{}units.h5'.format(n_units))

    # return all models
    return model, encoder_model, decoder_model


def predict_sequence(infenc, infdec, in_seq, n_output, target_token_ind,
                     reverse_target_char_ind, max_decoder_seq_len):
    # encode
    state = infenc.predict(in_seq)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(n_output)]).reshape(1, 1, n_output)
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_ind['\t']] = 1.
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)

        # Sample a token
        sampled_token_index = np.argmax(yhat[0, 0, :])
        sampled_char = reverse_target_char_ind[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        # target_seq = np.array([0.0 for _ in range(n_output)]).reshape(1, 1, n_output)
        # target_seq[0, 0, sampled_token_index] = 1.
        target_seq = yhat

        # Update states
        state = [h, c]

    return decoded_sentence


def main(input_params):
    current_mode = input_params.mode
    batch_size = input_params.batch_size  # Batch size for training.
    epochs = input_params.epochs  # Number of epochs to train for.
    num_units = input_params.num_units  # dimensionality of the output space
    num_samples_inf = input_params.num_inf_samples  # Number of samples for inferencing
    datafilename = input_params.datapath
    shuffle_data = input_params.shuffle_data

    # Path to the data txt file on disk.
    # data_path = os.path.realpath(os.getcwd()) + datafilename
    num_samples = input_params.num_trn_samples  # Number of samples to train on.
    train_indices = range(0, num_samples)
    test_indices = range(num_samples, min(num_samples + num_samples_inf, args.num_lines))
    input_texts, target_texts, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length,\
        max_decoder_seq_length, input_token_index, target_token_index, encoder_input_data,\
        decoder_input_data, decoder_target_data, reverse_input_char_index,\
        reverse_target_char_index, input_texts_test, target_texts_test, encoder_test_data, \
        decoder_test_data, decoder_test_target_data = prepare_data(datafilename, train_indices,
                                                                   test_indices, shuffle=shuffle_data)

    if current_mode == 'train':
        print("TRAINING====================================")
        model, encoder_model, decoder_model = define_models(num_encoder_tokens, num_decoder_tokens, num_units)

        # Run training
        early_stop = EarlyStopping(monitor='val_loss', verbose=1)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=batch_size, callbacks=[early_stop],
                  epochs=epochs,
                  validation_split=0.2, shuffle=True)

        # Save all models
        os.makedirs('models', exist_ok=True)
        model.save('models/seq2seq_model_{}units.h5'.format(num_units))
        encoder_model.save('models/seq2seq_encoder_model_{}units.h5'.format(num_units))
        decoder_model.save('models/seq2seq_decoder_model_{}units.h5'.format(num_units))

    elif current_mode == 'infer':
        print("INFERENCE====================================")
        model, encoder_model, decoder_model = load_defined_models(num_units)
        in_texts = []
        actual_texts = []
        predicted_texts = []
        for seq_index in range(len(test_indices)):
            # Take one sequence (part of the training test)
            # for trying out decoding.
            input_seq = encoder_test_data[seq_index: seq_index + 1]
            if len(input_seq) <= num_encoder_tokens:
                decoded_sentence = predict_sequence(encoder_model, decoder_model, input_seq, num_decoder_tokens,
                                                    target_token_index, reverse_target_char_index,
                                                    max_decoder_seq_length)

                actual_texts.append(target_texts_test[seq_index])
                in_texts.append(input_texts_test[seq_index])
                predicted_texts.append(decoded_sentence)
                print('INFERENCE SAMPLE {}===================================='.format(seq_index))
                print('Input sentence:', input_texts_test[seq_index])
                print('Decoded sentence:', decoded_sentence)
        # Bleu Scores
        print('Bleu-1: %f' % corpus_bleu(actual_texts, predicted_texts, weights=(1.0, 0, 0, 0)))
        print('Bleu-2: %f' % corpus_bleu(actual_texts, predicted_texts, weights=(0.5, 0.5, 0, 0)))
        print('Bleu-3: %f' % corpus_bleu(actual_texts, predicted_texts, weights=(0.3, 0.3, 0.3, 0)))
        print('Bleu-4: %f' % corpus_bleu(actual_texts, predicted_texts, weights=(0.25, 0.25, 0.25, 0.25)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, dest='mode', default='infer',
                        choices=['train', 'infer'],
                        help='Either "train" for training or "infer" for inference')
    parser.add_argument("--batch", type=int, dest='batch_size', default=64,
                        help='Batch size')
    parser.add_argument("--epochs", type=int, dest='epochs', default=100,
                        help='Number of epochs')
    parser.add_argument("--units", type=int, dest='num_units', default=256,
                        help='Number of LSTM units')
    parser.add_argument("--datapath", type=Path, dest='datapath', default='data/fra_shuffled.txt',
                        help='Path to data [default: data/fra_shuffled.txt]')
    parser.add_argument("--shuffle_data", action='store_true',
                        help="Shuffle input data and save")
    parser.add_argument("--train_size", type=int, dest='num_trn_samples', default=10000,
                        help='Number of samples for training')
    parser.add_argument("--infer_size", type=int, dest='num_inf_samples', default=10000,
                        help='Number of samples for training')
    args = parser.parse_args()

    # Only shuffle data for training
    args.shuffle_data = args.shuffle_data if args.mode == 'train' else False
    args.datapath = os.path.join(os.path.realpath(os.getcwd()), str(args.datapath))
    args.num_lines = sum(1 for line in io.open(args.datapath, 'r', encoding='utf-8'))

    main(args)