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

def calculate_loss(x):
	"""compute the cost as norm y"""
	print('x is:', x)
	model = load_model('model_sigmoid_extended.h5')
	session = k.get_session()
	# y = model.predict(np.reshape(x,(1,6)))
	[y0,y1,y2,y3,y4,y5,y6] = np.asarray(model.predict(np.reshape(x,(1,6))))
	y = np.concatenate((y0,y1,y2,y3,y4,y5,y6), axis = 1)
	loss = np.linalg.norm(y)
	print('The loss is:', loss)
	return loss
	
def calculate_gradient(x):
	model = load_model('model_sigmoid_extended.h5')
	session = k.get_session()
	x = np.reshape(x,(1,6))
	# y = model.predict(x)
	[y0,y1,y2,y3,y4,y5,y6] = np.asarray(model.predict(np.reshape(x,(1,6))))
	y = np.concatenate((y0,y1,y2,y3,y4,y5,y6), axis = 1)
	dy_dx = np.zeros((7,6))
	for i_check in range(7):
		# dy_dx_i = session.run(tf.gradients(model.output[0,i_check], model.input), feed_dict={model.input: x})
		dy_dx_i = session.run(tf.gradients(model.output[i_check], model.input), feed_dict={model.input: x})
		dy_dx[i_check,:] = np.asarray(dy_dx_i)
	
	dL_dy = y/np.linalg.norm(y)
	dL_dx = np.dot(dL_dy,dy_dx)
	print('The norm of the gradient is:', np.linalg.norm(dL_dx))
	return np.asarray(dL_dx)
	
if __name__ == '__main__':
	
	x0 = np.array([0.1, 0.75, 0.75, 0.15, 0.5, 0.15])
	bounds = Bounds([0,0,0,0,0,0],[1,1,1,1,1,1])
	res = minimize(calculate_loss, x0, method='SLSQP', jac = calculate_gradient,  bounds=bounds,
               options={'ftol': 1e-4, 'disp': True})
	min_tx = np.asarray([0.5e-07, 0.5e-07,0.05,0.05,0.25,0.25])
	max_tx = np.asarray([10e-07, 10e-07,0.49,0.49,1.5,1.5])
	diff_tx = max_tx - min_tx
	final_tx = np.multiply(diff_tx,np.asarray(res.x)) + min_tx
	print('Final values are',final_tx)
	model = load_model('model_sigmoid.h5')
	session = k.get_session()
	print('Final predicted errors are', model.predict(np.reshape(res.x,(1,6))))
	# final_properties = gradient_descent(session,model)
	# print('The final properties are:', final_properties)
	
	# print('Grad full is', grad_evaluated)
	
