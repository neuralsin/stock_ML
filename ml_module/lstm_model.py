# ml_module/lstm_model.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

class LSTMModel:
    def __init__(self, input_shape=(30, 6), model_name="lstm_stock.h5"):
        self.input_shape = input_shape
        self.model_path = os.path.join(MODEL_DIR, model_name)
        self.model = self._build_or_load_model()

    def _build_or_load_model(self):
        if os.path.exists(self.model_path):
            return load_model(self.model_path)
        
        model = Sequential()
        model.add(LSTM(128, input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Predict next price change or direction
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        if X_val is not None:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        self.save_model()
        return history

    def predict(self, X):
        """Return predicted next price/direction"""
        X = np.array(X)
        if len(X.shape) == 2:
            X = X.reshape((1, X.shape[0], X.shape[1]))
        return self.model.predict(X, verbose=0)

    def save_model(self):
        self.model.save(self.model_path)

    def load_model(self):
        self.model = load_model(self.model_path)
        return self.model
