import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers

from keras.metrics import MeanSquaredError

import keras

import json






def build_model(dropout = 0.2):

    metrics_model = Sequential()
    metrics_model.add(Dense(64, activation='relu', input_dim=64, kernel_initializer='normal'))
    metrics_model.add(Dropout(dropout))
    metrics_model.add(Dense(32, activation='relu', kernel_initializer='normal'))
    metrics_model.add(Dropout(dropout))
    metrics_model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    metrics_model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.1)))
    return metrics_model


def string_to_array(str):
    array = np.array([int(char) for char in str], dtype=np.uint8)
    array = array.reshape((8, 8))
    return array

def get_num_pieces(board, num):
    counter = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == num:
                counter+=1
    return counter

def evaluate(board):
    player_pieces = get_num_pieces(board, 1)
    player_kings = get_num_pieces(board, 2)
    computer_pieces = get_num_pieces(board, 3)
    computer_kings = get_num_pieces(board, 4)
    return player_pieces - computer_pieces + (player_kings * 4 - computer_kings * 4)


def scale_board(board):
    board = board.astype(np.float16)
    for i in range(len(board)):
        for j in range(len(board[i])):
            board[i][j] = board[i][j]/4
    return board


def prepare_data():
    file_path = "checkers_test_dict1.json"
    with open(file_path, 'r') as json_file:
        dict = json.load(json_file)

    X = []
    y = []
    for i in dict:
        board = string_to_array(i)
        X.append(board)
        y.append(dict[i][0])

    X = np.array(X,dtype=np.int8)
    y = np.array(y)
    X = X.reshape(-1,64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(model,X_train, X_test, y_train, y_test, batch_size=200, epochs=4, lr=0.001):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error', metrics=[MeanSquaredError()])
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test))










model = build_model()
#model = load_model('test_model.h5')
X_train, X_test, y_train, y_test = prepare_data()
z = train_model(model, X_train, X_test, y_train, y_test)
model.save("_model.h5")




