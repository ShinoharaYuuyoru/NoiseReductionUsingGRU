# Code by ShYy

# This script uses ShYy dataset, to create a Noise Added ShYy-DS-NA.

import os, random
import numpy as np
import scipy.io.wavfile as wav


# Mix Audio and Noise as 75%Audio and 25%Noise
def mix_audio(data, noise):
    mix = np.add(0.75*data, 0.25*noise)
    out = np.array(mix)

    return out


# Set a start point from front 15s of the noise audios. The human voice waves are all shorter than 15s.
def create_noise_piece(noisedata, data):
    datalength = len(data)
    startpoint = random.randrange(0, 720000)         # 30s noise has 1440000 sampled points at 48kHz

    outputdata = noisedata[startpoint:startpoint + datalength]

    return outputdata


# ----- Main Function Start -----

# Directories
wavsDir = os.getcwd() + "/Wavs/"
noiseDir = os.getcwd() + "/Noises/"
noiseAddedDir = os.getcwd() + "/Training/NoiseAdded/"
# humanVoiceDir = os.getcwd() + "/Training/HumanVoices/"

# Noise File Name
whiteNoise = "WhiteNoise.wav"
brownianNoise = "BrownianNoise.wav"
pinkNoise = "PinkNoise.wav"

# The length of the noise file. Three noise file are all 30s.
os.chdir(noiseDir)
tempNoiseData = wav.read(whiteNoise)
noiseLength = len(tempNoiseData)

# Get Noises
wNoiseRate, wNoise = wav.read(whiteNoise)
bNoiseRate, bNoise = wav.read(brownianNoise)
pNoiseRate, pNoise = wav.read(pinkNoise)

# Define Mixture of audio
WhiteNoiseMix = []
BrownianNoiseMix = []
PinkNoiseMix = []

# File Counter for Debugging
fileCounter = 0

# Enter /Wavs/ dir to start mixing processing
os.chdir(wavsDir)

# The wave files are combined in only 1 human voice channel.
# For each wave file, use the channel and sum it with three types of noises.
for fileName in os.listdir(wavsDir):
    if fileName.endswith(".wav"):
        # Read the wave file
        wavFileRate, wavFile = wav.read(fileName)

        # # Use the right channel
        # rightChannel = wavFile[:, 1]
        # humanVoice = np.array(rightChannel)

        # # Normalize human voice sample. Gain from the other channel, BGM channel.
        # humanVoicePeak = max(abs(humanVoice))
        # wavFilePeak = np.iinfo(wavFile.dtype).max
        # gain = float(wavFilePeak)/humanVoicePeak
        # humanVoiceNormalized = np.array(humanVoice * gain)

        # Create Mixtures
        print("Mixing " + fileName + " with Noises...")
        WhiteNoiseMix = np.array(mix_audio(wavFile, create_noise_piece(wNoise, wavFile)))
        BrownianNoiseMix = np.array(mix_audio(wavFile, create_noise_piece(bNoise, wavFile)))
        PinkNoiseMix = np.array(mix_audio(wavFile, create_noise_piece(pNoise, wavFile)))

        # Write mixture audio files in the training directory.
        os.chdir(noiseAddedDir)
        fName, extendName = os.path.splitext(fileName)
        wav.write(fName + "_wnoise" + extendName, wavFileRate, WhiteNoiseMix.astype(np.int16))
        wav.write(fName + "_bnoise" + extendName, wavFileRate, BrownianNoiseMix.astype(np.int16))
        wav.write(fName + "_pnoise" + extendName, wavFileRate, PinkNoiseMix.astype(np.int16))

        # # Write human voice audio to the directory for computing with Noise Added audios
        # os.chdir(humanVoiceDir)
        # wav.write(fName + "_voice" + extendName, wavFileRate, humanVoiceNormalized.astype(np.int16))

        # End and back to the Waves directory
        print("Finished Processing: " + fileName)
        os.chdir(wavsDir)

        # Counter++
        fileCounter = fileCounter + 1

print("Total Processed: " + str(fileCounter) + " file(s).")