
import os
import io
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "best_EfficientNetB2 (1).keras"
)

IMG_SIZE = (260, 260)

class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


def format_class_name(class_name):
    name = class_name.replace("Tomato___", "")
    name = name.replace("_", " ")

    if "Spider mites" in name:
        return "Spider Mites / Two-Spotted Spider Mite"

    if name.lower() == "healthy":
        return "Healthy"

    return name


# --------------------------------------------------
# 2. LOAD MODEL ONCE
# --------------------------------------------------

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

print("Loading tomato disease model...")
model = tf.keras.models.load_model(
    MODEL_PATH, compile=False
)

print("Model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        f"Model has {model.output_shape[-1]} outputs, "
        f"but {len(class_names)} class names are defined."
    )


# --------------------------------------------------
# 3. FLASK SERVER
# --------------------------------------------------

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return "Tomato disease AI server is running."


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "error": "No image received."
            }), 400

        image_bytes = request.files["image"].read()

        # Decode the ESP32-CAM JPEG as an RGB image.
        img = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Match tomato_leaf_detection.py.
        img = img.resize(IMG_SIZE)

        # Same preprocessing as your desktop script:
        # float image, RGB, no division by 255.
        img_array = np.asarray(
            img, dtype=np.float32
        )
        img_array = np.expand_dims(
            img_array, axis=0
        )

        raw_predictions = model.predict(
            img_array, verbose=0
        )[0]

        # Match your existing score conversion logic.
        if (
            np.any(raw_predictions < 0)
            or np.any(raw_predictions > 1)
            or not np.isclose(
                np.sum(raw_predictions), 1.0, atol=0.01
            )
        ):
            shifted = (
                raw_predictions - np.max(raw_predictions)
            )
            exp_values = np.exp(shifted)
            predictions = exp_values / np.sum(exp_values)
        else:
            predictions = raw_predictions

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        disease_name = format_class_name(predicted_class)
        confidence = float(predictions[predicted_index])

        status = (
            "HEALTHY"
            if "healthy" in predicted_class.lower()
            else "POSSIBLE DISEASE / PEST DETECTED"
        )

        top_indices = np.argsort(predictions)[::-1][:3]

        top3 = [
            {
                "class": format_class_name(class_names[int(i)]),
                "confidence": float(predictions[int(i)])
            }
            for i in top_indices
        ]

        print(
            f"Prediction: {disease_name} "
            f"({confidence * 100:.2f}%)"
        )

        return jsonify({
            "predicted_class": predicted_class,
            "disease_name": disease_name,
            "predicted_class_index": predicted_index,
            "confidence": confidence,
            "status": status,
            "top3": top3
        })

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
