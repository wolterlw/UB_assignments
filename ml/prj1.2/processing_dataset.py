import numpy as np
import pandas as pd

def import_data(file_path):
    data_target = []
    data_input = []
    with open(file_path, 'r') as f:
        while True:
            # Splitting our dataset per colums based on the space separator
            data_str = f.readline().split(' ')
            if not data_str[0]:
                break
            # Target is the first column
            data_target.append(float(data_str[0]))
            # Features are columns 2 to 48, where we take only the values
            data_input.append([float(a.split(':')[1]) for a in data_str[2:48]])

    x = np.array(data_input)
    y = np.array(data_target).reshape((-1, 1))

    return x, y

file_path = r'./Querylevelnorm.txt'
x, y = import_data(file_path)
print("X shape is", x.shape, "\nY shape is", y.shape)

n, m = x.shape

# Data partition
n_train = int(0.8 * n)
n_valid = int((n - n_train) / 2)

x_train = x[:n_train, :]
x_valid = x[n_train:n_train + n_valid, :]
x_test = x[n_train + n_valid:, :]

y_train = y[:n_train, :]
y_valid = y[n_train:n_train + n_valid, :]
y_test = y[n_train + n_valid:, :]

pd.DataFrame(x_train).to_csv('./x_train.csv')
pd.DataFrame(y_train).to_csv('./y_train.csv')
pd.DataFrame(x_test).to_csv('./x_test.csv')
pd.DataFrame(y_test).to_csv('./y_test.csv')