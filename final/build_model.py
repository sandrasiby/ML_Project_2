import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, np_utils
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

NUM_FEATURES = 6

def build_model():
	'''Keras API neural network model'''	
	
	# Input layer
	main_input = Input(shape=(NUM_FEATURES,), name='sigmoid')
	
	activation_function = 'sigmoid'
	
	# Hidden layers
	hidden_layer = Dense(40, activation=activation_function)(main_input)  # Hidden layer 1
	hidden_layer = Dense(30, activation=activation_function)(hidden_layer)# Hidden layer 2
			
	# Output layer
	''' Several separate layers in order to allow for individual loss weights ''' 
	o1 = Dense(1, kernel_initializer='normal')(hidden_layer)
	o2 = Dense(1, kernel_initializer='normal')(hidden_layer)
	o3 = Dense(1, kernel_initializer='normal')(hidden_layer)
	o4 = Dense(1, kernel_initializer='normal')(hidden_layer)
	o5 = Dense(1, kernel_initializer='normal')(hidden_layer)
	o6 = Dense(1, kernel_initializer='normal')(hidden_layer)
	o7 = Dense(1, kernel_initializer='normal')(hidden_layer)
	
	# Initialize the model
	model = Model(inputs=[main_input], outputs=[o1,o2,o3,o4,o5,o6,o7])
	
	# Assign the optimizer and the loss type, along with the metric to be displayed
	model.compile(optimizer='adam', loss='mean_squared_error',loss_weights=[0.1, 0.1, 0.1, 0.5, 0.6, 0.7, 1], metrics=['mse'])
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

def standardize_minmax(tx):
	''' Standardize data using minmax with user specified min and max 
		This helps with the predictions later in the other files.'''
	
	min_tx = np.asarray([0.5e-07, 0.5e-07,0.05,0.05,0.5,0.5])
	max_tx = np.asarray([10e-07, 10e-07,0.49,0.49,1.5,1.5])
	tx = (tx - min_tx)/(max_tx-min_tx)	
	
	return tx

def print_error_statistics(err):
	''' Function that prints all the error statistics for each individual output '''
	
	for i in range(7):
		err_y = err[:,i]
		print('%%%%%%%%%%%%%%%%%%%%% Error statistics for output y',i,' %%%%%%%%%%%%%%%%%%%%%%%')
		print('No. of samples with < 25% error =', len(err_y[err_y <= 25]))
		print('No. of samples with < 10% error =', len(err_y[err_y <= 10]))
		print('No. of samples with < 5% error =', len(err_y[err_y <= 5]))
		print('No. of samples with < 1% error =', len(err_y[err_y <= 1]))
	
		print('Maximum error overall', err_y.max(), '%')
		print('SD of error ', err_y.std(), '%')
		print('Mean of error ', err_y.mean(), '%')
		print('Median of error ', np.median(err_y), '%')			
	
if __name__ == '__main__':
	
	fv = "featureVec"
	ov = "objectiveVec"
	
	# Get the features and labels
	features, labels = get_features_labels(fv, ov)
	
	# Split the data
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.01) # Use 0.01 for the optimization model
	
	# Standardize the test and train data using min-max standardization
	X_train = standardize_minmax(X_train)
	X_test = standardize_minmax(X_test);
	
	# Train the neural network	
	model = build_model()
	model.fit([X_train], [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4],y_train[:,5],y_train[:,6]], epochs=2000, batch_size=10)
	
	# Get predictions for the test samples
	[y0,y1,y2,y3,y4,y5,y6] = np.asarray(model.predict(X_test))
	y_pred = np.concatenate((y0,y1,y2,y3,y4,y5,y6), axis = 1)

	# Manually check the error on the test samples
	err = abs(y_pred - y_test)*100 / y_test	
	print_error_statistics(err)
	
	# Save the ANN model	
	filepath = 'model.h5'
	model.save(filepath)
	
