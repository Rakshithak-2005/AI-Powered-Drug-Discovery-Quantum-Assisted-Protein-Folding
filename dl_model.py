import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

def train_dl(X_train, y_train, X_test):
    """
    Trains a neural network and returns the model and predictions
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(0.001), loss='mse')
    
    model.fit(X_train, y_train, 
              epochs=50, 
              batch_size=32, 
              verbose=0,
              validation_split=0.2)
    
    y_pred = model.predict(X_test).flatten()
    return model, y_pred