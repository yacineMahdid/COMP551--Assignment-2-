# COMP551--Assignment-2- Language Classification

## Neural Network

Steps to run:

	0. Copy x_train, y_train and x_test into the folder
	
	1. Run 'preprocess_nn.py'
		This will generate a .npz file with the data used for our neural network.

	2. Run 'nn.py'
		This will train the network.

	Alternatively, if present, you can run 'nn.py' directly, using the precompiled .npz file.

## Prerequisites and Recommendations

Prerequisites:
	- TensorFlow for Python 2.7 or 3.5(Windows) with GPU support
	- Keras for Python 2.7
	- Scikit-learn
	- All relevant libraries

Recommendations:
	For a larger number of epochs, it is highly recommended to use the following:
	- Computer with good CPU
	- GPU computing such as CUDA and CuDNN

	Results reported were obtained using CUDA with NVIDIA GeForce GTX970 GPU
