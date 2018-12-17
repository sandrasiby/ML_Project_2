import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from keras import backend as k
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from itertools import cycle
from keras.models import load_model

NUM_FEATURES = 6

def run_model(X_train, X_test, y_train, y_test, tag):

	# Standardize the test and train data using min-max standardization
	X_train, mean_train, std_train = standardize_training_minmax(X_train)
	X_test = standardize_test_minmax(X_test, mean_train, std_train);
	print(len(X_train), len(y_train), len(X_test), len(y_test))

	# Train the neural network	
	model = build_model()
	model.fit([X_train], [y_train], epochs=1500, batch_size=10)
	score = model.evaluate(X_test, y_test)
	model.save("model_fold_" + tag + ".h5")
	print(score)

	# Get predictions
	y_pred = model.predict(X_test)
	#print("y pred", y_pred)

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

	return err.mean()

def cross_validate(features, labels, folds=10):

	results = []
	feature_sets = np.split(features, folds)
	label_sets = np.split(labels, folds)

	for i in range(0, folds):
		X_train = (feature_sets[i:] + feature_sets[:i])[:-1]
		X_train = np.concatenate(X_train)
		X_test = (feature_sets[i:] + feature_sets[:i])[-1]
		y_train = (label_sets[i:] + label_sets[:i])[:-1]
		y_train = np.concatenate(y_train)
		y_test = (label_sets[i:] + label_sets[:i])[-1]
		results.append(run_model(X_train, X_test, y_train, y_test, str(i)))
	
	return results

def build_model():
	
	''' Keras API Model '''	
	# Input layer
	main_input = Input(shape=(NUM_FEATURES,), name='input')
	
	activation_function = 'sigmoid'
	# Hidden layers
	hidden_layer = Dense(32, activation=activation_function)(main_input)  # Hidden layer 1
	hidden_layer = Dense(32, activation=activation_function)(hidden_layer)# Hidden layer 2 
	hidden_layer = Dense(32, activation=activation_function)(hidden_layer)# Hidden layer 3
	hidden_layer = Dense(20, activation=activation_function)(hidden_layer)# Hidden layer 4
	hidden_layer = Dense(20, activation=activation_function)(hidden_layer)# Hidden layer 5
	
	# Output layer 
	output_layer = Dense(7, kernel_initializer='normal')(hidden_layer)
	
	# Initialize the model
	model = Model(inputs=[main_input], outputs=[output_layer])
	
	# Assign the optimizer and the loss type, along with the metric to be displayed
	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
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
	#min_tx[:2] = 1e-07#
	max_tx = tx.max(axis=0)
	tx = (tx - min_tx)/(max_tx-min_tx)	
	return tx, min_tx, max_tx
	
def standardize_test_minmax(tx,min,max):
	''' Remove outliers and standardize the data'''
	# Perform an initial standardization of the data 
	tx = (tx - min)/(max - min)	
	return tx	

def calculate_loss(y):
    """compute the cost as norm y"""
    
    loss = np.linalg.norm(y)
    dL_dy = y/loss
	
    return loss, dL_dy
	
def gradient_descent(session,model,gamma = 0.01, threshold = 1e-06):
	init_x = [0.5] * 6
	loss = 100
	x = init_x
	x = np.reshape(x,(1,6))
	print("In GD, X", x)
	iter = 0
	#grad_order = np.asarray([1e-06, 1e-06,0.1,0.1,1,1])
	while ((loss > threshold) and (iter < 30)):
		
		
		y = model.predict(x)
		# loss, dL_dy = calculate_loss(y)
		loss = y[0,6]
		print("x is", x)
		print("The loss is", loss)
		# print('dL_dy is', dL_dy)
		dy_dx = calculate_gradient(session,model,y,x)
		# print('dy_dx is', dy_dx)
		# dL_dx = np.dot(dL_dy,dy_dx)
		dL_dx = dy_dx
		# print('dL_dx is', dL_dx)
		
		
		# print("The gradient is", dy_dx)
		x = x - gamma * dL_dx
		#x = x - gamma *np.multiply(grad_order, dL_dx)
		iter +=1
	return x

def check_weights():

	tag = "model_fold_"
	for i in range(0, 10):
		print("Fold", i)
		model = load_model(tag + str(i) + ".h5")
		ct = 0
		for layer in model.layers:
			print("Layer", ct)
			weights = layer.get_weights()
			if len(weights) > 0:
				print("Max:", np.max(weights[0]), "Min:", np.min(weights[0]), "Mean:", np.mean(weights[0]), "Std:", np.std(weights[0]))
				aw = np.absolute(weights[0])
				print("Max:", np.max(aw), "Min:", np.min(aw), "Mean:", np.mean(aw), "Std:", np.std(aw))
			ct += 1
		#break
	
def calculate_gradient(session,model,y,x):
	dy_dx = np.zeros((1,6))
	for i_check in range(1):
		dy_dx_i = session.run(tf.gradients(model.output[0,6], model.input), feed_dict={model.input: x})
		dy_dx[i_check,:] = np.asarray(dy_dx_i)
	return np.asarray(dy_dx)
	
if __name__ == '__main__':

	fv = "featureVec_Refine"
	ov = "ObjectiveVec_Refine"
	
	Get the features and labels
	features, labels = get_features_labels(fv, ov)
	results = cross_validate(features, labels)

	print("All mean errors")
	for item in results:
		print(item)

	print("Mean of errors:", np.mean(results))
	#check_weights()