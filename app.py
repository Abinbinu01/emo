import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# load trained LSTM model + label classes
model = tf.keras.models.load_model("model.h5")
classes = np.load("classes.npy", allow_pickle=True)

# load scaler if you used normalization in training (recommended)
scaler = joblib.load("scaler.save")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    data = request.json
    ks = data["keystrokes"]

    key_intervals = []
    pauses = 0
    prev_key_time = None

    first_time = None
    last_time = None
    char_count = len(ks)

    for k in ks:
        t = k["keydown"]

        # first and last time for typing speed
        if first_time is None:
            first_time = t
        last_time = t

        # key interval calculation
        if prev_key_time is not None:
            interval = t - prev_key_time
            key_intervals.append(interval)

            # pause count (> 1s gap)
            if interval > 1000:
                pauses += 1

        prev_key_time = t

    if len(key_intervals) < 2:
        return jsonify({"prediction": "Not enough typing data"})

    # ---------- FEATURE CALCULATION ----------
    avg_interval = float(np.mean(key_intervals))

    # typing speed = characters per minute
    duration_minutes = max((last_time - first_time) / 60000, 0.0001)
    typing_speed = float(char_count / duration_minutes)

    # ---------- HARD RULE OVERRIDE ----------
    # High speed + lots of pauses = STRESSED
    if pauses >= 12 and typing_speed >= 65:
        return jsonify({"prediction": "Stressed"})

    # ---------- ML MODEL PREDICTION ----------
    features = np.array([[avg_interval, pauses, typing_speed]])

    # scale features (same as training)
    features = scaler.transform(features)

    # reshape for LSTM (samples, timesteps, features)
    features = np.expand_dims(features, axis=1)

    pred = model.predict(features)
    idx = int(np.argmax(pred))
    emotion = str(classes[idx])

    return jsonify({
        "prediction": emotion,
        "avg_interval": round(avg_interval, 2),
        "pauses": pauses,
        "typing_speed": round(typing_speed, 2)
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



