import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Concatenate, Input, concatenate
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

NUM_FEATURES = 7776 # These are the total degrees of freedom per geometry. 
NUM_GEOMETRIES = 7

def build_model():
	''' Keras API Model '''
	
	# Obtain all the input geometry vectors (7)
	input_1 = Input(shape=(NUM_FEATURES,), name='input_1')
	input_2 = Input(shape=(NUM_FEATURES,), name='input_2')
	input_3 = Input(shape=(NUM_FEATURES,), name='input_3')
	input_4 = Input(shape=(NUM_FEATURES,), name='input_4')
	input_5 = Input(shape=(NUM_FEATURES,), name='input_5')
	input_6 = Input(shape=(NUM_FEATURES,), name='input_6')
	input_7 = Input(shape=(NUM_FEATURES,), name='input_7')
	print('shape of input_1 is', K.int_shape(input_1))
	
	# First set that combines information from each input itself
	combine_1 = Dense(1000, activation='relu')(input_1)
	combine_1 = Dense(500, activation='relu')(combine_1)
	combine_1 = Dense(50, activation='relu')(combine_1)
	combine_2 = Dense(1000, activation='relu')(input_2)
	combine_2 = Dense(500, activation='relu')(combine_2)
	combine_2 = Dense(50, activation='relu')(combine_2)
	combine_3 = Dense(1000, activation='relu')(input_3)
	combine_3 = Dense(500, activation='relu')(combine_3)
	combine_3 = Dense(50, activation='relu')(combine_3)
	combine_4 = Dense(1000, activation='relu')(input_4)
	combine_4 = Dense(500, activation='relu')(combine_4)
	combine_4 = Dense(50, activation='relu')(combine_4)
	combine_5 = Dense(1000, activation='relu')(input_5)
	combine_5 = Dense(500, activation='relu')(combine_5)
	combine_5 = Dense(50, activation='relu')(combine_5)
	combine_6 = Dense(1000, activation='relu')(input_6)
	combine_6 = Dense(500, activation='relu')(combine_6)
	combine_6 = Dense(50, activation='relu')(combine_6)
	combine_7 = Dense(1000, activation='relu')(input_7)
	combine_7 = Dense(500, activation='relu')(combine_7)
	combine_7 = Dense(50, activation='relu')(combine_7)
	print('shape of combine_1 is', K.int_shape(combine_1))
	# Put all the above layers together
	x = concatenate([combine_1,combine_2,combine_3,combine_4,combine_5,combine_6,combine_7])
	print('shape of x is', K.int_shape(x))
	# First hidden layer that combines all the above data
	hidden_1 = Dense(32, activation='relu')(x)
	print('shape of hidden_1 is', K.int_shape(hidden_1))
	hidden_1 = Dense(32, activation='relu')(hidden_1)
	hidden_1 = Dense(32, activation='relu')(hidden_1)
	hidden_1 = Dense(32, activation='relu')(hidden_1)
	hidden_1 = Dense(32, activation='relu')(hidden_1)
	hidden_1 = Dense(32, activation='relu')(hidden_1)
	output_layer = Dense(6, activation='relu')(hidden_1)
	# o1 = Dense(1, activation='relu')(hidden_1)
	# o2 = Dense(1, activation='relu')(hidden_1)
	# o3 = Dense(1, activation='relu')(hidden_1)
	# o4 = Dense(1, activation='relu')(hidden_1)
	# o5 = Dense(1, activation='relu')(hidden_1)
	# o6 = Dense(1, activation='relu')(hidden_1)
	
	model = Model(inputs=[input_1,input_2,input_3,input_4,input_5,input_6,input_7], outputs=[output_layer])
	print('shape of outputs is', K.int_shape(output_layer))
	model.compile(optimizer='adam', loss='mean_absolute_error',  metrics=['mse'])
	print(model.metrics_names)

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

def standardize_training(tx):
	''' Remove outliers and standardize the data'''
	# Perform an initial standardization of the data 
	mean_tx = np.mean(tx,axis=0)
	tx = tx - mean_tx
	std_tx = np.std(tx,axis=0)
	tx = tx / std_tx
	
	return tx, mean_tx, std_tx
	
if __name__ == '__main__':

	fv = "featureVec"
	ov = "objectiveVec"

	fv = "featureVec_Refine"
	ov = "objectiveVec_Refine"
	
	print("BEGIN: File Read")
	labels, gv_110 = get_features_labels(fv, "geomVec_P110")
	labels, min, max = standardize_training(labels)
	labels = labels * 100
	gv_10, gv_20 = get_features_labels("geomVec_P10", "geomVec_P20")
	gv_30, gv_40 = get_features_labels("geomVec_P30", "geomVec_P40")
	gv_50, gv_70 = get_features_labels("geomVec_P50", "geomVec_P70")
	print("END: File Read")
	features = np.concatenate((gv_10,gv_20,gv_30,gv_40,gv_50,gv_70,gv_110),axis = 1)
	print("BEGIN: Split")
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.02)
	print(len(X_train), len(y_train), len(X_test), len(y_test))
	print("END: Split")
	model = build_model()
	input_1 = X_train[:,:NUM_FEATURES]
	input_2 = X_train[:,NUM_FEATURES:2*NUM_FEATURES]
	input_3 = X_train[:,2*NUM_FEATURES:3*NUM_FEATURES]
	input_4 = X_train[:,3*NUM_FEATURES:4*NUM_FEATURES]
	input_5 = X_train[:,4*NUM_FEATURES:5*NUM_FEATURES]
	input_6 = X_train[:,5*NUM_FEATURES:6*NUM_FEATURES]
	input_7 = X_train[:,6*NUM_FEATURES:7*NUM_FEATURES]
	
	
	test_1 = X_test[:,:NUM_FEATURES]
	test_2 = X_test[:,NUM_FEATURES:2*NUM_FEATURES]
	test_3 = X_test[:,2*NUM_FEATURES:3*NUM_FEATURES]
	test_4 = X_test[:,3*NUM_FEATURES:4*NUM_FEATURES]
	test_5 = X_test[:,4*NUM_FEATURES:5*NUM_FEATURES]
	test_6 = X_test[:,5*NUM_FEATURES:6*NUM_FEATURES]
	test_7 = X_test[:,6*NUM_FEATURES:7*NUM_FEATURES]
	
	
	# model.fit([input_1,input_2,input_3,input_4,input_5,input_6,input_7], 
		# [y_train[:,0], y_train[:,1], y_train[:,2], y_train[:,3], y_train[:,4], y_train[:,5]], epochs=1, batch_size=400)
	model.fit([input_1,input_2,input_3,input_4,input_5,input_6,input_7], [y_train], epochs=50, batch_size=400)
	# score = model.evaluate([test_1,test_2,test_3,test_4,test_5,test_6,test_7], [y_test[:,0], y_test[:,1], y_test[:,2], y_test[:,3], y_test[:,4], y_test[:,5]])
	print('shape of X_test is', np.shape(X_test))
	print('Shape of test_1', np.shape(test_1))
	y_pred = model.predict([test_1,test_2,test_3,test_4,test_5,test_6,test_7])
	print('Shape of y_pred', np.shape(y_pred))
	print('y_pred is',y_pred)
	print('y_test is',y_test)
	# err = (y_pred - y_test)*100 / y_test
	# print(err)
	#print accuracy_score(y_test, y_pred1.transpose())

