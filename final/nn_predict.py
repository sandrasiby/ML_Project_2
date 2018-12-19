import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from keras import backend as k
from keras.models import load_model
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.optimize import minimize, Bounds

NUM_FEATURES = 6


def standardize_training_minmax(tx):
	''' Remove outliers and standardize the data'''
	# Perform an initial standardization of the data 
	# min_tx = tx.min(axis=0)
	min_tx = np.asarray([1e-07, 1e-07,0.05,0.05,0.5,0.5])
	max_tx = np.asarray([10e-07, 10e-07,0.49,0.49,1.5,1.5])
	# max_tx = tx.max(axis=0)
	tx = (tx - min_tx)/(max_tx-min_tx)	
	return tx, min_tx, max_tx

if __name__ == '__main__':
	
	model = load_model('model_sigmoid.h5')
	session = k.get_session()
	
	#e1 = 1.85e-07
	#e2 = 3.057e-07
	#nu1 = 0.25
	#nu2 = 0.435
	#m1 = 1.08
	#m2 = 0.675
	
	#e1 = 1.35e-07
	#e2 = 1.057e-07
	#nu1 = 0.15
	#nu2 = 0.235
	#m1 = 1.25
	#m2 = 0.75	
	
	e1 = 3.3293623e-07
	e2 = 1e-07
	nu1 = 0.49
	nu2 = 0.05
	m1 = 1.5
	m2 = 0.5223

	x = np.array([e1,e2,nu1,nu2,m1,m2])
	
	tx ,min,max= standardize_training_minmax(x)
	[y0,y1,y2,y3,y4,y5,y6] = np.asarray(model.predict(np.reshape(tx,(1,6))))
	y_pred = np.concatenate((y0,y1,y2,y3,y4,y5,y6), axis = 1)
	print('Final predicted errors are', y_pred)
	# final_properties = gradient_descent(session,model)
	# print('The final properties are:', final_properties)
	
	# print('Grad full is', grad_evaluated)
	
