import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def train_dl(X_train, y_train, X_test, epochs=30, batch_size=32, random_state=42):
    """
    Trains a deep neural network regression model on molecular features
    and returns the fitted model and predictions.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,)
    X_test : array-like of shape (n_samples_test, n_features)
    epochs : int, default=30
    batch_size : int, default=32
    random_state : int, default=42

    Returns
    -------
    model : Sequential
    y_pred : np.ndarray
    """
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    X_tr = np.asarray(X_train, dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.float32)
    X_te = np.asarray(X_test, dtype=np.float32)

    n_features = X_tr.shape[1]

    model = Sequential([
        Input(shape=(n_features,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    model.fit(
        X_tr,
        y_tr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.2
    )

    y_pred = model.predict(X_te, verbose=0).flatten()
    return model, np.asarray(y_pred, dtype=float)
