import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.metrics import MeanSquaredError
from keras import backend as K
import wandb
import keras
import json
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.login()

#https://github.com/wandb/examples/blob/master/colabs/keras/Keras_param_opti_using_sweeps.ipynb
#https://blog.paperspace.com/building-a-checkers-gaming-agent-using-neural-networks-and-reinforcement-learning/

def string_to_array(str):
    array = np.array([int(char) for char in str], dtype=np.uint8)
    array = array.reshape((8, 8))
    return array

file_path = "checkers_test_dict1.json"
with open(file_path, 'r') as json_file:
    dict = json.load(json_file)

X = []
y = []
for i in dict:
    board = string_to_array(i)
    X.append(board)
    y.append(dict[i][0])


X = np.array(X, dtype=np.int8)
y = np.array(y)
X = X.reshape(-1, 64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = []
y = []
dict = {}





def build_model(activation1,activation2,activation3,activation4,dropout = 0.1, num = 0, regu = 0.1):
    if num == 0:
        metrics_model = Sequential()
        metrics_model.add(Dense(64, activation=activation1, input_dim=64, kernel_initializer='normal'))
        metrics_model.add(Dropout(dropout))
        metrics_model.add(Dense(32, activation=activation2, kernel_initializer='normal'))
        metrics_model.add(Dropout(dropout))
        metrics_model.add(Dense(16, activation=activation3, kernel_regularizer=regularizers.l2(regu)))
        metrics_model.add(Dense(1, activation=activation4, kernel_regularizer=regularizers.l2(regu)))
    if num == 1:
        metrics_model = Sequential()
        metrics_model.add(Dense(32, activation='relu',input_dim=64, kernel_initializer='normal'))
        metrics_model.add(Dropout(dropout))
        metrics_model.add(Dense(64, activation=activation1, kernel_initializer='normal'))
        metrics_model.add(Dropout(dropout))
        metrics_model.add(Dense(32, activation=activation2, kernel_initializer='normal'))
        metrics_model.add(Dropout(dropout))
        metrics_model.add(Dense(16, activation=activation3, kernel_regularizer=regularizers.l2(regu)))
        metrics_model.add(Dense(1, activation=activation4, kernel_regularizer=regularizers.l2(regu)))
    if num == 2:
        metrics_model = Sequential()
        metrics_model.add(Dense(32, activation='relu', input_dim=64, kernel_initializer='normal'))
        metrics_model.add(Dropout(dropout))
        metrics_model.add(Dense(32, activation='relu', kernel_initializer='normal'))
        metrics_model.add(Dropout(dropout))
        metrics_model.add(Dense(64, activation=activation1, kernel_initializer='normal'))
        metrics_model.add(Dropout(dropout))
        metrics_model.add(Dense(32, activation=activation2, kernel_initializer='normal'))
        metrics_model.add(Dropout(dropout))
        metrics_model.add(Dense(16, activation=activation3, kernel_regularizer=regularizers.l2(regu)))
        metrics_model.add(Dense(1, activation=activation4, kernel_regularizer=regularizers.l2(regu)))


    return metrics_model



def train_model(model, batch_size=64, epochs=10, lr=1e-3, optimizer='adam', log_freq=10):



    model.compile(optimizer=get_optimizer(lr, optimizer), loss='mean_squared_error', metrics=[MeanSquaredError()])

    wandb_callbacks = [
        WandbMetricsLogger(log_freq=log_freq),
        WandbModelCheckpoint(filepath=r"C:\Users\User\Documents\pythonProject\checkpoints")
    ]
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              callbacks=wandb_callbacks)


def get_optimizer(lr=1e-3, optimizer="adam"):
    "Select optmizer between adam and sgd with momentum"
    if optimizer.lower() == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if optimizer.lower() == "sgd":
        return keras.optimizers.SGD(learning_rate=lr)



sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'mean_squared_error',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'dropout': {
          'values': [0.1, 0.2, 0.3]
        },
    'epochs': {
        'values': [5,10,20,30]
        },
    'learning_rate': {
        'distribution': 'uniform',
        'min': 0.0001,
        'max': 0.1
      },
    'batch_size': {
        'values': [16, 32,64]
      },
    'num': {
          'values': [0, 1, 2]
        },
    'activation1': {
          'values': ['relu', 'sigmoid']
        },
    'activation2': {
          'values': ['relu', 'sigmoid']
        },
    'activation3': {
          'values': ['relu', 'sigmoid']
        },
    'activation4': {
          'values': ['relu', 'sigmoid']
        },
    'regu': {
        'distribution': 'uniform',
        'min': 0.001,
        'max': 0.2
        }
    }

sweep_config['parameters'] = parameters_dict
sweep_config['early_terminate'] = {"type": "hyperband", "min_iter": 3}



def sweep_train(config_defaults=None):
    # Initialize wandb with a sample project name
    with wandb.init(config=config_defaults):  # this gets over-written in the Sweep

        # Specify the other hyperparameters to the configuration, if any
        wandb.config.architecture_name = "Sequential"
        wandb.config.dataset_name = "checkers"

        # initialize model
        model = build_model(wandb.config.activation1,
                            wandb.config.activation2,
                            wandb.config.activation3,
                            wandb.config.activation4,
                            wandb.config.dropout,
                            wandb.config.num,
                            wandb.config.regu)

        train_model(model,
              wandb.config.batch_size,
              wandb.config.epochs,
              wandb.config.learning_rate,
              wandb.config.optimizer)




sweep_id = wandb.sweep(sweep_config, project="sweeps-keras")
wandb.agent(sweep_id, function=sweep_train, count=50)

# https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu
