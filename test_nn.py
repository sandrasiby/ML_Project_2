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

def build_model():

	main_input = Input(shape=(NUM_FEATURES,), name='input')
	x = Dense(32, activation='relu')(main_input)
	o1 = Dense(1, activation='relu')(x)
	o2 = Dense(1, activation='relu')(x)
	o3 = Dense(1, activation='relu')(x)
	o4 = Dense(1, activation='relu')(x)
	o5 = Dense(1, activation='relu')(x)
	o6 = Dense(1, activation='relu')(x)
	o7 = Dense(1, activation='relu')(x)

	model = Model(inputs=[main_input], outputs=[o1, o2, o3, o4, o5, o6, o7])
	model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], metrics=['accuracy'])
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

if __name__ == '__main__':

	fv = "featureVec"
	ov = "objectiveVec"

	features, labels = get_features_labels(fv, ov)
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.02)
	print len(X_train), len(y_train), len(X_test), len(y_test)
	model = build_model()
	model.fit([X_train], [y_train[:,0], y_train[:,1], y_train[:,2], y_train[:,3], y_train[:,4], y_train[:,5], y_train[:,6]],
          epochs=1, batch_size=32)
	score = model.evaluate(X_test, [y_test[:,0], y_test[:,1], y_test[:,2], y_test[:,3], y_test[:,4], y_test[:,5], y_test[:,6]])
	print(score)
	y_pred = np.array(model.predict(X_test))
	
	y_pred1 = y_pred[:, :, 0]
	err = y_pred1.transpose() - y_test / y_test
	print err
	#print accuracy_score(y_test, y_pred1.transpose())

