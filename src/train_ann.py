from src.data_loader import load_data
from src.preprocessing import add_rul, prepare_features

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

train_df = load_data("data/train_FD001.txt")

train_df = add_rul(train_df)

X, y, scaler = prepare_features(train_df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("ANN MAE:", mae)

model.save("models/ann_model.h5")