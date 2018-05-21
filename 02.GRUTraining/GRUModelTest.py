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
def formatSrcFilename(filename):
    return filename[:len(filename) - 11] + "_voice.wav"

def formatOutputFilename(filename):
    return filename[:len(filename) - 11] + "_output.wav"

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
    final_data = np.zeros([len(batch), max_run_total, stft_size, sequence_length], dtype=np.float32)
    true_time = np.zeros([len(batch), max_run_total], dtype=np.int32)

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
graphPath = os.getcwd() + "/TF_Checkpoints/FINAL.ckpt"

# Number of test files
testFileNum = 0

# File List
testNAFileList = []         # Test Dataset. Noise Added File List.
srcHVFileList = []          # Source Human Voice File List.
outputFileList = []         # Output File List

# File Repository
testNARateRepository = []
testNADataRepository = []
srcHVRateRepository = []
srcHVDataRepository = []

norm_factor = (1.0 / 32768.0)         # Let data map to -1 ~ 1 range for LSTM process

# Walk all test NA files to File List and File Repository.
for root, _, files in os.walk(testData):
    files = sorted(files)
    testFileNum = len(files)

    for f in files:
        if f.endswith(".wav"):
            testNAFileList.append(f)
            rate, data = wav.read(os.path.join(root, f))
            testNARateRepository.append(rate)
            testNADataRepository.append(data * norm_factor)

srcHVFileList = list(map(formatSrcFilename, testNAFileList))
outputFileList = list(map(formatOutputFilename, testNAFileList))

# Walk all source HV files to File Repository.
for root, _, files in os.walk(humanVoice):
    files = sorted(files)

    for f in files:
        if(f.endswith(".wav")):
            for name in srcHVFileList:
                if f == name:
                    rate, data = wav.read(os.path.join(root, f))
                    srcHVRateRepository.append(rate)
                    srcHVDataRepository.append(data * norm_factor)

# STFT Process Variables, also used in LSTM
sequence_length = 100
stft_size = 1024
batch_size = 1          # Set 1 for process 1 Wav file a time.
number_of_layers = 3

# Tensorflow vars + Graph and LSTM Params
input_data = tf.placeholder(tf.float32, [None, sequence_length, stft_size])
# clean_data = tf.placeholder(tf.float32, [None, sequence_length, stft_size])
sequence_length_tensor = tf.placeholder(tf.int32, (None))

# TF Graph Definition
gru_cell = tf.contrib.rnn.GRUCell(stft_size, kernel_initializer = tf.zeros_initializer(dtype = tf.float32))
# gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, dtype = tf.float32, output_keep_prob = 0.5)            # Cancel Dropout
stacked_gru = tf.contrib.rnn.MultiRNNCell([gru_cell] * number_of_layers, state_is_tuple=True)
init_state = stacked_gru.zero_state(batch_size, tf.float32)
rnn_outputs, final_state = tf.nn.dynamic_rnn(stacked_gru, input_data, sequence_length=sequence_length_tensor, initial_state=init_state, time_major=False)
# mse_loss = tf.losses.mean_squared_error(rnn_outputs, clean_data)
# train_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdagradDAOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss)
saver = tf.train.Saver()

# Initialize TF Graph and Restore the Graph
init_op = tf.global_variables_initializer()  # initialize_all_variables()
gpu_options = tf.GPUOptions(allow_growth = True)            # Set session GPU using growing.
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
sess.run(init_op)
saver.restore(sess, graphPath)
print("\t***** TF GRAPH RESTORED *****")

# Start Processing
for idx in range(testFileNum):
    nowNAFile = []
    nowNAFile.append(testNADataRepository[idx])

    # Get NA stft repository.
    nowNAData_STFT, sequenceLengthID, maxLength = sequentialized_spectrum(nowNAFile)

    # Get Time Steps.
    maxTimeSteps = len(nowNAData_STFT[0])

    # Define outputData List to contain rnn_outputs_value.
    outputData = np.zeros([1,  maxTimeSteps, stft_size, sequence_length])           # Transpose, [0, 1, 3, 2]

    for timeStep in range(maxTimeSteps):
        feed_dict = {
            input_data : nowNAData_STFT[:, timeStep, :],
            sequence_length_tensor : sequenceLengthID[:, timeStep]
        }
        final_state_value, rnn_outputs_value = sess.run([final_state, rnn_outputs], feed_dict=feed_dict)

        rnn_outputs_value = np.transpose(rnn_outputs_value, [0, 2, 1])
        outputData[0][timeStep] = rnn_outputs_value

    # Define outputData_STFT, link outputData List by timeStep in 1 dimension.
    outputData_STFT = np.zeros([stft_size, maxLength])
    beginTime = 0
    endTime = 0
    for timeStep in range(maxTimeSteps):
        if(timeStep < maxTimeSteps - 1):
            endTime = beginTime + sequence_length
            outputData_STFT[:, beginTime : endTime] = outputData[0, timeStep, :, :]
        else:
            endTime = beginTime + int(sequenceLengthID[0, timeStep])
            outputData_STFT[:, beginTime : endTime] = outputData[0, timeStep, :, 0 : (endTime - beginTime)]

        beginTime = beginTime + sequence_length

    # Compute ISTFT
    _, outputData_ISTFT = signal.istft(outputData_STFT, fs=testNARateRepository[0], nperseg=stft_size, input_onesided = False)

    outputData_ISTFT = ((outputData_ISTFT / norm_factor).real) / 0.75
    outputData_ISTFT = outputData_ISTFT.astype(np.int16)

    wav.write(modelOutput + outputFileList[idx], testNARateRepository[idx], outputData_ISTFT)
    print("Index: " + str(idx))
    print("\tOutput File: " + str(outputFileList[idx]))