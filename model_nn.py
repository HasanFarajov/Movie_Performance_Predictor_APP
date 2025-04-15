# Deep Learning (Keras + Scikeras) training script

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib

# Dataset yüklə
df = pd.read_csv("D:/ASOIU MASTER/II kurs II sem/Fuzzy/midterm/movies/movies1.csv")
df = df.dropna(subset=["Gross"])

# X və y
X = df[["Budget", "Runtime", "Rating", "Rating Count"]]
y = df["Gross"]

# Train/test bölməsi
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Neural Net qurucu
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Regression çıxışı
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Pipeline
nn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nn', KerasRegressor(model=build_model, epochs=100, batch_size=16, verbose=0))
])

# Təlim
nn_pipeline.fit(X_train, y_train)

# Modeli saxla
joblib.dump(nn_pipeline, "movie_gross_predictor_nn.pkl")