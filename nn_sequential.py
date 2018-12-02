import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

NUM_FEATURES = 6


# define base model
def build_model():
	# create model
	model = Sequential()
	activation_function = 'relu'
	model.add(Dense(20, input_dim=NUM_FEATURES, kernel_initializer='normal', activation = activation_function))
	model.add(Dense(20, kernel_initializer='normal', activation = activation_function))
	model.add(Dense(20, kernel_initializer='normal', activation = activation_function))
	model.add(Dense(20, kernel_initializer='normal', activation = activation_function))
	model.add(Dense(20, kernel_initializer='normal', activation = activation_function))
	model.add(Dense(7, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
	return model
	
	
def get_features_labels(fv, ov):

	features = []
	with open(fv) as f:
		lines = f.readlines()
		for line in lines:
			a = line.split(",")
			features.append([float(x) for x in a])

	labels = []
	with open(ov) as f:
		lines = f.readlines()
		for line in lines:
			a = line.split(",")
			labels.append([float(x) for x in a])

	return np.array(features), np.array(labels)

def standardize_training(tx):
	''' Remove outliers and standardize the data'''
	# Perform an initial standardization of the data 
	mean_tx = np.mean(tx,axis=0)
	tx = tx - mean_tx
	std_tx = np.std(tx,axis=0)
	tx = tx / std_tx
	
	return tx, mean_tx, std_tx

def standardize_training_minmax(tx):
	''' Remove outliers and standardize the data'''
	# Perform an initial standardization of the data 
	min_tx = tx.min(axis=0)
	max_tx = tx.max(axis=0)
	tx = (tx - min_tx)/(max_tx-min_tx)	
	return tx, min_tx, max_tx
	
def standardize_test_minmax(tx,min,max):
	''' Remove outliers and standardize the data'''
	# Perform an initial standardization of the data 
	tx = (tx - min)/(max - min)	
	return tx
	
def standardize_test(tx,mean,std):
	''' Remove outliers and standardize the data'''
	# Perform an initial standardization of the data 
	tx = tx - mean
	tx = tx / std
	
	return tx	
	
if __name__ == '__main__':

	fv = "featureVec_Refine"
	ov = "objectiveVec_Refine"

	features, labels = get_features_labels(fv, ov)
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.2)
	print(np.shape(X_train), np.shape(y_train), np.shape(X_test), np.shape(y_test))
	X_train, mean_train, std_train = standardize_training_minmax(X_train)
	X_test = standardize_test_minmax(X_test, mean_train, std_train);
	model = build_model()
	model.fit(X_train, y_train, epochs=4000, batch_size=400)
	
	score = model.evaluate(X_test, y_test)
	# print('The score is:', score)
	# print('X first row is : ', X_test[0,:])
	y_pred = model.predict(X_test)
	# print('y_pred is', y_pred)
	# print('y_test is', y_test)
	
	err = abs(y_pred - y_test)*100 / y_test
	err_max_sample = err.max(1)
	print('Maximum error in each test sample', err_max_sample)
	err_less_than_25 = err_max_sample[err_max_sample <= 25]
	err_less_than_10 = err_max_sample[err_max_sample <= 10]
	err_less_than_5 = err_max_sample[err_max_sample <= 5]
	print('No. of samples with < 25% error =', len(err_less_than_25))
	print('No. of samples with < 10% error =', len(err_less_than_10))
	print('No. of samples with < 5% error =', len(err_less_than_5))
	# print('error in each sample', err)
	print('Maximum error overall', err.max(), '%')
	print('SD of error ', err.std(), '%')
	print('Mean of error ', err.mean(), '%')
	print('Median of error ', np.median(err), '%')
	#print accuracy_score(y_test, y_pred1.transpose())



