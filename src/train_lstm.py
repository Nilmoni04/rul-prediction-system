from src.data_loader import load_data
from src.preprocessing import add_rul, prepare_features

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import numpy as np

train_df = load_data("data/train_FD001.txt")

train_df = add_rul(train_df)

X, y, scaler = prepare_features(train_df)

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32
)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("LSTM MAE:", mae)

model.save("models/lstm_model.h5")