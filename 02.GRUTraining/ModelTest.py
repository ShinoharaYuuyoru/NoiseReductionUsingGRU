# This script aims to test the model using the Test Dataset seperated from source Dataset by CreateTestDataset.py
# Code by ShYy, 2018.

import scipy
import scipy.signal as signal
import numpy as np
import os
import random
import sys
import scipy.io.wavfile as wav
import tensorflow as tf
import math

# Get the source Human Voice file names by Noise Added file names.
def formatFilename(filename):
    return filename[:len(filename) - 11] + "_voice.wav"

def sequentialized_spectrum(batch):
    # Get maximum length of batch
    t = []
    t_vec = []
    Sxx_Vec = []
    for each in batch:
        _, t, Sxx_Vec_Temp = signal.stft(each, fs=testNARateRepository[0], nperseg=stft_size, return_onesided = False)
        t_vec.append(t)
        Sxx_Vec.append(Sxx_Vec_Temp)
    maximum_length = findMaxlen(t_vec)

    max_run_total = int(math.ceil(float(maximum_length) / sequence_length))
    final_data = np.zeros([len(batch), max_run_total, stft_size, sequence_length])
    true_time = np.zeros([len(batch), max_run_total])

    # Read in a file and compute spectrum
    # for batch_idx, each_set in enumerate(batch):
    for batch_idx, Sxx in enumerate(Sxx_Vec):
        # f, t, Sxx = signal.stft(each_set, fs=rate_repository[0], nperseg=stft_size, return_onesided = False)

        # Magnitude and Phase Spectra
        Mag = Sxx.real
        t = t_vec[batch_idx]
        # Phase = Sxx.imag

        # Break up the spectrum in sequence_length sized data
        run_full_steps = float(len(t)) / sequence_length
        run_total = int(math.ceil(run_full_steps))

        # Run a loop long enough to break up all the data in the file into chunks of sequence_size
        for step in range(run_total):

            begin_point = step * sequence_length
            end_point = begin_point + sequence_length

            m, n = Mag[:, begin_point:end_point].shape

            # Store each chunk sequentially in a new array, accounting for zero padding when close to the end of the file
            if n == sequence_length:
                final_data[batch_idx, step, :, :] = np.copy(Mag[:, begin_point:end_point])
                true_time[batch_idx, step] = n
            else:
                final_data[batch_idx, step, :, :] = np.copy(create_final_sequence(Mag[:, begin_point:end_point], sequence_length))
                true_time[batch_idx, step] = n

    final_data = np.transpose(final_data, (0, 1, 3, 2))

    return final_data, true_time, maximum_length

def findMaxlen(data_vec):
    max_ = 0
    for each in data_vec:
        if len(each) > max_:
            max_ = len(each)
    return max_

def create_final_sequence(sequence, max_length):
    a, b = sequence.shape
    extra_len = max_length - b
    null_mat = np.zeros((len(sequence), extra_len), dtype=np.float32)
    sequence = np.concatenate((sequence, null_mat), axis=1)
    return sequence

# Directories
humanVoice = os.getcwd() + "/Training/HumanVoices/"
testData = os.getcwd() + "/Testing/NoiseAdded/"
modelOutput = os.getcwd() + "/Testing/ModelOutput/"

# Number of test files
testFileNum = 0

# File List
testNAFileList = []         # Test Dataset. Noise Added File List.
srcHVFileList = []          # Source Human Voice File List.

# File Repository
testNARateRepository = []
testNADataRepository = []
srcHVRateRepository = []
srcHVDataRepository = []

# Walk all test NA files to File List and File Repository.
for root, _, files in os.walk(testData):
    files = sorted(files)
    testFileNum = len(files)

    for f in files:
        if f.endswith(".wav"):
            testNAFileList.append(f)
            rate, data = wav.read(os.path.join(root, f))
            testNARateRepository.append(rate)
            testNADataRepository.append(data)

srcHVFileList = list(map(formatFilename, testNAFileList))

# Walk all source HV files to File Repository.
for root, _, files in os.walk(humanVoice):
    files = sorted(files)

    for f in files:
        if(f.endswith(".wav")):
            for name in srcHVFileList:
                if f == name:
                    rate, data = wav.read(os.path.join(root, f))
                    srcHVRateRepository.append(rate)
                    srcHVDataRepository.append(data)

# STFT Process Variables, also used in LSTM
sequence_length = 100
stft_size = 1024
norm_factor = (1.0 / 32768.0)         # Let data map to -1 ~ 1 range for LSTM process
batch_size = 1

# Get NA stft repository
testNADataRepository_STFT, sequenceLengthID, maxLength = sequentialized_spectrum(testNADataRepository * norm_factor)


# Tensorflow vars + Graph and LSTM Params
input_data = tf.placeholder(tf.float32, [None, sequence_length, stft_size])
# clean_data = tf.placeholder(tf.float32, [None, sequence_length, stft_size])
sequence_length_tensor = tf.placeholder(tf.int32, (None))

# TF Graph Definition
lstm_cell = tf.contrib.rnn.BasicLSTMCell(stft_size, forget_bias = 1.0, state_is_tuple = True)
# stacked_lstm = tf.contrib.rnn.MultiRNNCell([[lstm_cell] for i in number_of_layers])
init_state = lstm_cell.zero_state(batch_size, tf.float32)
rnn_outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, input_data, sequence_length=sequence_length_tensor, initial_state=init_state, time_major=False)
# mse_loss = tf.losses.mean_squared_error(rnn_outputs, clean_data)
# train_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdagradDAOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss)
saver = tf.train.Saver()