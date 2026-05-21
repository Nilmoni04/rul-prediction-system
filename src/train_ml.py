from src.data_loader import load_data
from src.preprocessing import add_rul, prepare_features

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import joblib


train_df = load_data("data/train_FD001.txt")

train_df = add_rul(train_df)

X, y, scaler = prepare_features(train_df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("ML Model MAE:", mae)

joblib.dump(model, "models/ml_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")