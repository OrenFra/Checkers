import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras import backend as K
from keras.metrics import MeanSquaredError
from keras.models import load_model
import keras
import pickle
import json



#https://github.com/wandb/examples/blob/master/colabs/keras/Keras_param_opti_using_sweeps.ipynb
#https://blog.paperspace.com/building-a-checkers-gaming-agent-using-neural-networks-and-reinforcement-learning/



def build_model(dropout = 0.2):

    metrics_model = Sequential()
    metrics_model.add(Dense(64, activation='relu', input_dim=64, kernel_initializer='normal'))
    metrics_model.add(Dropout(dropout))
    metrics_model.add(Dense(32, activation='relu', kernel_initializer='normal'))
    metrics_model.add(Dropout(dropout))
    metrics_model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    metrics_model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.1)))
    return metrics_model

    '''model = Sequential()#2D model
    model.add(layers.Conv2D(32, (3, 3), padding='SAME', activation='relu', input_shape=(8, 8, 1)))
    model.add(layers.MaxPooling2D((2, 2), padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3), padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2),padding='SAME'))
    model.add(layers.Conv2D(64, (3, 3),padding='SAME', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), padding='SAME'))
    model.add(layers.Conv2D(1, (3, 3), padding='SAME', activation='sigmoid'))
    model.add(layers.Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))'''

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

    '''X_train = X_train.reshape(-1, 64)
    X_test = X_test.reshape(-1, 64)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)'''
    return X_train, X_test, y_train, y_test


def train_model(model,X_train, X_test, y_train, y_test, batch_size=200, epochs=4, lr=0.001):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error', metrics=[MeanSquaredError()])
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test))





def data(X):# different presentation of the data in a binary way
    test_X = np.zeros((len(X), 320), dtype=int)
    for i in range(len(X)):
        current_X = np.zeros(320, dtype=int)
        for j in range(len(X[0])):
            current_X[(j*5) + (int)(X[i][j]*4)] = 1
        test_X[i] = current_X

    return test_X




model = build_model()
#model = load_model('test_model.h5')
X_train, X_test, y_train, y_test = prepare_data()
z = train_model(model, X_train, X_test, y_train, y_test)
model.save("all_data_model.h5")




'''for i in range(12):
    dropout = 0.1
    batch_size = 64
    epochs = 5
    lr = 0.001
    if i == 0:
        activation1 = activation2 = activation3 = activation4 = 'relu'
        lr = 0.000001
    if i == 1:
        activation1 = activation2 = activation3 = activation4 = 'sigmoid'
        epochs = 10
    if i == 2:
        activation1 = activation2 = activation3 = activation4 = 'softmax'
    if i == 3:
        activation1 = activation2 = activation3 = activation4 = 'tanh'
    if i == 4:
        activation1 = activation2 = activation3 = activation4 = 'selu'
    if i == 5:
        activation1 = activation2 = activation3 = activation4 = 'exponential'
    if i == 6:
        activation1 = activation2 = activation3 ='relu'
        activation4 = 'sigmoid'
    if i == 7:
        activation1 = activation2 = activation3 = 'relu'
        activation4 = 'sigmoid'
        dropout = 0.3
    if i == 8:
        activation1 = activation2 = activation3 = 'relu'
        activation4 = 'sigmoid'
        batch_size = 128
    if i == 9:
        activation1 = activation2 = activation3 = 'relu'
        activation4 = 'sigmoid'
        lr = 0.01
    if i == 10:
        activation1 = activation2 = activation3 = 'relu'
        activation4 = 'sigmoid'
        lr = 0.005
    if i == 11:
        activation1 = activation2 = activation3 = 'relu'
        activation4 = 'sigmoid'
        dropout = 0.01
    if i%3==0:
        X_train, X_test, y_train, y_test = prepare_data('0')
    if i%3==1:
        X_train, X_test, y_train, y_test = prepare_data('1')
    if i%3==2:
        X_train, X_test, y_train, y_test = prepare_data('2')
    model = build_model(activation1,activation2,activation3,activation4,dropout)
    print("activation1 = ",activation1,"activation2 = ",activation2,"activation3 = ",activation3,"activation4 = ",activation4,"dropout = ",dropout,"epochs = ",epochs,"learning rate = ",lr,"batch size = ",batch_size)
    z = train_model(model, X_train, X_test, y_train, y_test,batch_size,epochs,lr)'''



'''with open('model.pkl', 'wb') as file:#check if output is between 0 to 1
    pickle.dump(model, file)
model = pickle.load(open("model.pkl", "rb"))
scaler = MinMaxScaler()
x= np.load('X0.npy')
x = x.reshape(-1, 64)
x = x.astype(int)
X_train, X_test = train_test_split(x, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
counter = 0

predictions = model.predict(X_test)
for i in range(len(predictions)):
    if predictions[i] <0 or predictions[i] >1:
        counter+=1

print(counter)
print(counter/len(X_test))'''

# https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu
