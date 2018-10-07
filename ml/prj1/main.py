# beautified version of the code found in jupyter notebook
# for more comments and experiments look at notebook.ipynb

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, SGD, Adam


columns = ['Other', 'Fizz', 'Buzz', 'FizzBuzz']
optimizers = {
	'SGD': SGD,
	'RMSprop': RMSprop,
	'Adam': Adam
}

input_size = 10
output_size = 4

earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=0, patience=100, mode='min')

def decode(x):
	"""decodes strings from one-hot representation"""

	idx = np.argmax(x,axis=1)
	return [columns[z] for z in idx]

def fizz_buzz(x):
	"""Vectorized fizzBuzz implementation"""

	fizz = x % 3 == 0
	buzz = x % 5 == 0
	fizz_buzz = fizz & buzz
	other = ~(fizz | buzz)
	res = pd.DataFrame({
			'Other': other,
			'Fizz': fizz,
			'Buzz': buzz,
			'FizzBuzz': fizz_buzz,
		}, columns = columns, index=x).astype('uint8')
	res['Fizz'] -= res['FizzBuzz']
	res['Buzz'] -= res['FizzBuzz']
	return res

def enocode(x, dim=10):
	"""Converts digit to it's binary representation trimmed to (dim) bits"""

	return np.r_[[[i >> d & 1 for d in range(dim)] for i in x]]

def get_train_test(train_num, test_num):
	"""Creates train and test data
	(as out datasets are small - kept them in memory)."""

	X_tr = enocode(train_num)
	y_tr = fizz_buzz(train_num)

	X_ts = enocode(test_num)
	y_ts = fizz_buzz(test_num)

	data = {
		'X_tr': X_tr,
		'y_tr': y_tr,
		'X_ts': X_ts,
		'y_ts': y_ts,
	}
	return data

def get_model(hyper_params, optimizer='RMSprop', optimizer_params={'lr': 0.01}):
	"""Returns with a specified set of hyperparameters."""

	assert len(hyper_params['hidden_layer_nodes']) == hyper_params['num_hidden'], "specify layer size for each hidden layer"
	
	model = Sequential()
	
	model.add(Dense(hyper_params['hidden_layer_nodes'][0], input_dim=input_size))
	model.add(Activation(hyper_params['activation']))
	model.add(Dropout(hyper_params['drop_out']))
	
	for i in range(1,hyper_params['num_hidden']):
		model.add(Dense(hyper_params['hidden_layer_nodes'][i]))
		model.add(Activation(hyper_params['activation']))
		model.add(Dropout(hyper_params['drop_out']))
	
	model.add(Dense(output_size))
	model.add(Activation('softmax'))
	
	optimizer = optimizers[optimizer](**optimizer_params)
	
	model.compile(optimizer=optimizer,
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])

	return model

def fit_model(model, data, num_epochs=1000, batch_size=32):
	"""Trains (model) on (data) for (num_epochs) with a specified (batch_size)."""

	model.fit(data['X_tr'],
			  data['y_tr'],
			  validation_split = 0.2,
			  epochs = num_epochs,
			  batch_size = batch_size,
			  callbacks = [earlystopping_cb],
			  verbose = False)

def perform_experiment(model):
	"""Performs experiment and generates output."""

	test_vals = np.arange(1,101)
	data = get_train_test(np.array([]), test_vals)
	label = decode(data['y_ts'].values)

	pred = model.predict(data['X_ts'])
	pred_label = decode(pred)

	header = pd.DataFrame({
		'input': ['UBID','personNumber'],
		'label': ['vliunda', '50291163'],
		'predicted_label': ['','']
		}, columns=['input','label','predicted_label'])

	test_res = pd.DataFrame({
		'input': test_vals,
		'label': label,
		'predicted_label': pred_label
		}, columns=['input','label','predicted_label']) 

	return pd.concat([header, test_res]).reset_index(drop=True)


if __name__ == '__main__':

	# hyperparameters optimized with Bayesian optimization
	# to see the experiment see jupyterNotebook
	hyper_params = {
		'drop_out':  0.339,
		'hidden_layer_nodes': [245, 154, 67],
		'num_hidden': 3,
		'activation': 'tanh',
	}
	optimizer_params = {
		'lr': 0.00815
	}

	data = get_train_test(np.arange(101,1001), np.arange(1,101))
	model = get_model(hyper_params, optimizer_params=optimizer_params)
	fit_model(model, data)

	perform_experiment(model).to_csv('./output.csv')


