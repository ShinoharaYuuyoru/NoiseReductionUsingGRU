# Code By adityatb at https://github.com/adityatb/noise-reduction-using-rnn
# GRU method.
# Maintain by ShYy, 2018.

import scipy
import scipy.signal as signal
import numpy as np
import os, random, sys
import scipy.io.wavfile as wav
import tensorflow as tf
import math


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def formatFilename(filename):
    return filename[:len(filename) - 11] + "_voice.wav"


# Strip away the _xnoise.wav part of the filename, and append _voice.wav to obtain clean voice counterpart

def create_final_sequence(sequence, max_length):
    a, b = sequence.shape
    extra_len = max_length - b
    null_mat = np.zeros((len(sequence), extra_len), dtype=np.float32)
    sequence = np.concatenate((sequence, null_mat), axis=1)
    return sequence


def sequentialized_spectrum(batch):
    # Get maximum length of batch
    t = []
    t_vec = []
    Sxx_Vec = []
    for each in batch:
        _, t, Sxx_Vec_Temp = signal.stft(each, fs=rate_repository[0], nperseg=stft_size, return_onesided = False)
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


# ----------------- Begin Vars --------------------- #

# Training data directories
traindata = os.getcwd() + "/Training/NoiseAdded/"
voicedata = os.getcwd() + "/Training/HumanVoices/"
checkpoints = os.getcwd() + "/TF_Checkpoints/"

# NormConstant
norm_factor = (1 / 32768.0)

# Spectrogram Parameters
stft_size = 1024

# RNN Specs
sequence_length = 100
batch_size = 10
learning_rate = 0.0005
epochs = 250
number_of_layers = 3

# Tensorflow vars + Graph and GRU Params
input_data = tf.placeholder(tf.float32, [None, sequence_length, stft_size])
clean_data = tf.placeholder(tf.float32, [None, sequence_length, stft_size])
sequence_length_tensor = tf.placeholder(tf.int32, (None))

# Temp_data_variables
no_of_files = 0
temp_list = []
final_data = []
sequence_length_id = 0

# Repositories
file_repository = []
rate_repository = []
clean_repository = []

# Selected vectors
files_vec = []
clean_files_fin_vec = []
clean_files_vec = []

# Graph
gru_cell = tf.contrib.rnn.GRUCell(stft_size)
gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, output_keep_prob = 0.5)
stacked_gru = tf.contrib.rnn.MultiRNNCell([gru_cell] * number_of_layers, state_is_tuple=True)
init_state = stacked_gru.zero_state(batch_size, tf.float32)
rnn_outputs, final_state = tf.nn.dynamic_rnn(stacked_gru, input_data, sequence_length=sequence_length_tensor, initial_state=init_state, time_major=False)
mse_loss = tf.losses.mean_squared_error(rnn_outputs, clean_data)
# train_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(mse_loss)
# train_optimizer = tf.train.AdagradDAOptimizer(learning_rate).minimize(mse_loss)
train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss)
saver = tf.train.Saver()

# ------------------- Read all data to memory creating a repository of mixture and clean files --------------------- #

os.chdir(traindata)
# for file_iter in range(traindata):

# Buffer training data to memory for faster execution:
for root, _, files in os.walk(traindata):
    files = sorted(files)
    no_of_files = len(files)

    if batch_size > no_of_files:
        sys.exit("Error: batch_size cannot be more than number of files in the training directory")

    for f in files:
        if f.endswith(".wav"):
            temp_list.append(f)
            srate, data = wav.read(os.path.join(root, f))
            file_repository.append(data)
            rate_repository.append(srate)

# Generate a vector of file names that are clean files
clean_files_vec = list(map(formatFilename, temp_list))
# clean_files_vec = list(map(None, *clean_files_vec))

# Find clean files that correspond to data in file_repository and buffer clean voice data to memory
for root, _, files in os.walk(voicedata):
    files = sorted(files)
    for each in files:
        if each.endswith(".wav"):
            for name in clean_files_vec:
                if each == name:
                    srate2, data2 = wav.read(os.path.join(root, name))
                    # In Create Noise Adding Dataset, the NA audio is 0.75*Source + 0.25*Noise.
                    # So we need let clean data*0.75.
                    clean_repository.append(data2 * 0.75)

# ------------------- Step 1: Prepare data in batches and perform STFTs --------------------- #


# files_vec = []
run_epochs = int((no_of_files / batch_size) * epochs)

# Initialize TF Graph
init_op = tf.global_variables_initializer()  # initialize_all_variables()
gpu_options = tf.GPUOptions(allow_growth = True)            # Set session GPU using growing.
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
sess.run(init_op)

globalBatchLossSum = 0          # Sum of all batch losses
# globalStepsSum = 0          # Sum of all steps
lastCumulativeLossSum = 99999           # Last Cumulative Loss Sum.

for idx in range(int(run_epochs)):

    files_vec = []
    # clean_files_vec = []
    clean_files_fin_vec = []

    # Select batch_size no. of random number of files from file_repository and the corresponding clean files
    for file_iter in range(batch_size):
        i = random.randint(0, len(file_repository) - 1)
        files_vec.append(file_repository[i] * norm_factor)
        clean_files_fin_vec.append(clean_repository[i] * norm_factor)

    stft_batch, sequence_length_id, maximum_length = sequentialized_spectrum(files_vec)
    clean_voice_batch, sequence_length_id_clean, maximum_length_clean = sequentialized_spectrum(clean_files_fin_vec)

    # ------------------- Step 2: Feed Data to Placeholders, and then, Initialise, Train and Save the Graph  --------------------- #

    max_time_steps = stft_batch.shape[1]
    batchLossSum = 0            # Sum of batch losses in one index.

    for time_seq in range(max_time_steps):
        feed_dict = {
            input_data: stft_batch[:, time_seq, :, :],
            clean_data: clean_voice_batch[:, time_seq, :, :],
            sequence_length_tensor: sequence_length_id[:, time_seq]
        }
        _, loss_value, final_state_value, rnn_outputs_val = sess.run([train_optimizer, mse_loss, final_state, rnn_outputs], feed_dict=feed_dict)

        # print("Index " + str(idx + 1) + " in " + str(run_epochs))
        # print("\tOutput Min:\t" + str(np.min(rnn_outputs_val)))
        # print("\tClean Min:\t" + str(np.min(clean_voice_batch[:, time_seq, :, :])))
        # print("\tOutput Max:\t" + str(np.max(rnn_outputs_val)))
        # print("\tClean Max:\t" + str(np.max(clean_voice_batch[:, time_seq, :, :])))
        # print("\tBatch Loss:\t" + str(loss_value * 32768))          # Multiplied 32768 to show the batch losses obviously.
        batchLossSum = batchLossSum + loss_value

    print("Index " + str(idx + 1) + "/" + str(run_epochs) + " Batch Loss Sum:\t" + str(batchLossSum / norm_factor) + "\n")

    globalBatchLossSum = globalBatchLossSum + batchLossSum
    # globalStepsSum = globalStepsSum + max_time_steps

    if (int((idx + 1) % no_of_files) == 0):
        # All batch losses sum divide global steps to get Avg
        # cumulativLossAvg = globalBatchLossSum / globalStepsSum
        cumulativeLossSum = globalBatchLossSum
        print("\n\t\tCumulative epochs loss Sum in latest " + str(idx + 1) + " indexes:\t" + str(cumulativeLossSum / norm_factor))
        if(cumulativeLossSum < lastCumulativeLossSum):
            lastCumulativeLossSum = cumulativeLossSum            # If cumulative loss avg is smaller or equal to last avg, stay learning rate
        else:
            learning_rate = learning_rate / 5           # If cumulative loss avg is bigger than last avg, than change learning rate to 1/5
            lastCumulativeLossSum = cumulativeLossSum
            print("\n\t\tLearning Rate changed to: " + str(learning_rate))
        globalBatchLossSum = 0          # Initialize to 0, for next indexes batch loss calculation
        # globalStepsSum = 0

        os.chdir(checkpoints)
        saver.save(sess, './ssep_model.ckpt', global_step=idx)
        print("\t\tSaved checkpoint\n")
        os.chdir(traindata)

os.chdir(checkpoints)
saver.save(sess, './FINAL.ckpt')
print("Saved FINAL")
sess.close()