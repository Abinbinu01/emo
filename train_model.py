import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# -------- LOAD DATA --------
df = pd.read_csv("dataset/keystroke_features.csv")

X = df[["avg_key_interval", "total_pauses", "typing_speed"]].values
y = df["emotion"].values

# -------- LABEL ENCODE --------
le = LabelEncoder()
y = le.fit_transform(y)
np.save("classes.npy", le.classes_)

# -------- SCALE FEATURES (2-D) --------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# save scaler
joblib.dump(scaler, "scaler.save")

# -------- RESHAPE FOR LSTM (3-D) --------
# (samples, time_steps=1, features)
X = np.expand_dims(X, axis=1)

# -------- BUILD MODEL --------
model = Sequential()
model.add(LSTM(32, input_shape=(1, 3)))
model.add(Dense(4, activation="softmax"))

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# -------- TRAIN / TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# -------- TRAIN --------
model.fit(X_train, y_train, epochs=80, batch_size=8)

# -------- EVALUATE --------
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

# -------- SAVE MODEL --------
model.save("model.h5")
print("Model & scaler saved.")

test = np.array([[520, 10, 18]])  # clearly sad pattern
test = np.expand_dims(test, axis=1)

pred = model.predict(test)
print("Test prediction:", le.inverse_transform([np.argmax(pred)])[0])
