

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from tensorflow.keras.models import Sequential
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten
from colorama import Fore
def get_dnn_model(train_data, feature_columns, target_column):
    model_dnn = tf.keras.models.Sequential([
        tf.keras.layers.Dense(50, activation='relu', input_shape=(len(feature_columns),)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model_dnn.compile(optimizer='adam', loss='mean_squared_error')
    model_dnn.fit(train_data[feature_columns], train_data[target_column], epochs=100, batch_size=32, verbose=1)
    return model_dnn

def get_cnn_model(train_data, feature_columns, target_column):
    train_data.fillna(train_data.mean(), inplace=True)  # Remplacer les NaN par la moyenne
    model_cnn = tf.keras.models.Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(len(feature_columns), 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
    ])
    model_cnn.compile(optimizer='adam', loss='mean_squared_error')
    X_train_cnn = np.expand_dims(train_data[feature_columns].values, axis=2)  # Reshape pour CNN
    model_cnn.fit(X_train_cnn, train_data[target_column], epochs=100, batch_size=32, verbose=1)
    return model_cnn
def get_mlp_model(train_data, feature_columns, target_column):

# Define MLP model
    model_mlp = Sequential([
    Dense(50, activation='relu', input_shape=(len(feature_columns),)),
    Dense(50, activation='relu'),
    Dense(1)
    ])

    # Compile the model
    model_mlp.compile(optimizer='adam', loss='mean_squared_error')


    # Train the model
    model_mlp.fit(train_data[feature_columns], train_data[target_column], epochs=100, batch_size=32, verbose=1)

    return model_mlp



def get_lstm(train_data, feature_columns, target_column):
    # Obtain the DataFrames for training and validation

    model_lstm = tf.keras.models.Sequential([
        LSTM(50, activation='relu', input_shape=(len(feature_columns), 1)),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    # Reshape data for LSTM (add one dimension)
    X_train_lstm = np.expand_dims(train_data[feature_columns].values, axis=2)
    #X_valid_lstm = np.expand_dims(valid_data[feature_columns()].values, axis=2)

    # Train the model
    model_lstm.fit(X_train_lstm, train_data[target_column].values, epochs=100, batch_size=32, verbose=1)

    return model_lstm


def get_ann(train_data, feature_columns, target_column):
    train_features = np.random.rand(100, len(feature_columns))  # Exemple de données d'entraînement
    train_target = np.random.rand(100)  # Exemple de cible d'entraînement
    model_ann = Sequential([
        Dense(50, activation='relu', input_shape=(len(feature_columns),)),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model_ann.compile(optimizer='adam', loss='mean_squared_error')
    model_ann.fit(train_features, train_target, epochs=100, batch_size=32, verbose=1)
    return model_ann

#

