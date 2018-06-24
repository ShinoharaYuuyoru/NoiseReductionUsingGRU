# NoiseReductionUsingGRU  
Graduation Project Title: Noise Reduction Using GRU(RNN).  
  
This is my graduation project in BIT, 2018.  
  
The project based on Python 3.5 and TensorFlow 1.8.  
The project learning from [Noise Reduction using RNNs with Tensorflow](https://github.com/adityatb/noise-reduction-using-rnn) by adityatb.  
Thanks for adityatb's work!  
adityatb uses LSTM to build noise reduction model.  
  
With the development of RNN, I think GRU may be a better method.  
This graduation project uses [MIR-1K](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) dataset and try to use GRU to build noise reduction model.  
  
## Introduction  
This project includes 2 main part.  
1. CreateNoiseAddedDataset  
	This script will add 3 kinds of noises(Brownian, Pink and White) to the clean human voice.  
	If you have 1000 human voice files(MIR-1K has 1000 usable files), you will get 3000 noise added files.  
  
2. GRUTraining  
	This is a big part including 5 scripts.  
	1. CreateTestDataset.py  
		In last step, we have got 3000 files.  
		By this script, we randomly separate 1000 files to be used in Test process.  
  
	2. LSTMTestTraining.py  
		This script and the next script aim to check LSTM model's performance.  
		To see the GRU model works or not.  
		  
		adityatb used LSTM model, but the original code has a lot of problems:  
			>Python 2.x -> Python 3.x  
			>Array overflow problems  
			>Fourier trasformation using error  
			>No Learning Rate auto change  
			>...  
		I solved a lot. It is hard to introduce all at here.  
		Please refer to the history of git publishment.  
  
	3. ModelTest.py  
		This script will use the LSTM model trained in last script and execute 1000 test sounds.  
		To listen and compare the spectrum of [clear voice], [noise added sound] and [output] to see the performance.  
  
	4. GRUTraining.py  
		This script is modified by `LSTMTestTraining.py`.  
		Change LSTM to GRU in TensorFlow framwork is very easy.  
  
	5. GRUModelTest.py  
		This script is modified by `LSTMTestTraining.py`/  
	  
## Execute Steps  
*NOTICE:The detail of the implement, please read the code, refer to the modification history.*  
1. CreateNoiseAddedDataset  
	Download [MIR-1K](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) dataset. Decompress the folder `Wavfile`. We will get 1000 wav files.  
	Set all wav files into the `Wavs` folder.  
	Run `CreateNoiseAddedDataset.py`.  
	We will get 1000 human voice files in `./Training/HumanVoices`.  
	3000 noise added sounds in `./Training/NoiseAdded`.  
  
2. GRUTraining  
	1. CreateTestDataset  
		Cut the `./Training` folder into this part folder.  
		Run `CreateNoiseAddedDataset.py`.  
		We will get 1000 random chosen noise added files in `./Testing/NoiseAdded`.  
		And in `./Training/NoiseAdded` will 2000 files left.  
  
	2. GRUTraining  
		Run `GRUTraining.py`.  
		This script will train a GRU model from remaining 2000 files and corresponding pure human voice files.  
		Finally we will get TensorFlow checkpoint files in `./TFCheckpoints` folder.  
  
	3. GRUModelTest  
		Run `GRUModelTest.py`.  
		We will get less than 1000 files(because some noise added files correspond to the same pure human voice file) in `./Testing/ModelOutput`  
		You can test the model or do something else you like.  
  
***FINISH!***