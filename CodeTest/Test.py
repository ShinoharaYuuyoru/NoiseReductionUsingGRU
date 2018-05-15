import os
import scipy.io.wavfile as wav
import scipy.signal as signal

humanVoice = os.getcwd() + "/abjones_1_01_voice.wav"
whiteNoise = os.getcwd() + "/abjones_1_01_wnoise.wav"
brownNoise = os.getcwd() + "/abjones_1_01_bnoise.wav"
pinkNoise = os.getcwd() + "/abjones_1_01_pnoise.wav"

rate0, data0 = wav.read(humanVoice)
rate1, data1 = wav.read(whiteNoise)
rate2, data2 = wav.read(brownNoise)
rate3, data3 = wav.read(pinkNoise)

_, t0, _ = signal.stft(data0, fs = 16000, nperseg = 1024, return_onesided = True)
_, t1, _ = signal.stft(data1, fs = 16000, nperseg = 1024, return_onesided = True)
_, t2, _ = signal.stft(data2, fs = 16000, nperseg = 1024, return_onesided = True)
_, t3, _ = signal.stft(data3, fs = 16000, nperseg = 1024, return_onesided = True)

print("END")