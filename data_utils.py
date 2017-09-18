#16/09/2017
import numpy as np
import pandas as pd

def csv_to_numpy_array(train_path, test_path):
	print("loading training data")
	traindata = pd.read_csv(train_path, header= None, delim_whitespace=True).values
	#traindata = np.genfromtxt(train_path, dtype=None)

	print("loading test data")
	testdata = pd.read_csv(test_path, header= None, delim_whitespace=True).values
	#testdata = np.genfromtxt(test_path, dtype=None)

	return transform_data(traindata, testdata)

def transform_data(x_train, x_test):
	hot_classes = 10
	[trainX, trainY] = np.hsplit(x_train,[784])
	trainY = np.int_(trainY.reshape(-1))
	trainY = np.eye(hot_classes)[trainY] #one hot vector

	[testX, testY] = np.hsplit(x_test,[784])
	testY = np.int_(testY.reshape(-1))
	testY = np.eye(hot_classes)[testY] #one hot vector

	return trainX, trainY, testX, testY