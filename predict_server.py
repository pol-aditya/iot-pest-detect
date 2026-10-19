
import os
import io
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

HF_REPO_ID = "apol123/efficientnet_iot"
MODEL_FILENAME = "best_EfficientNetB2 (1).keras"

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
# 2. DOWNLOAD AND LOAD MODEL ONCE
# --------------------------------------------------

print("Downloading or locating model on Hugging Face...")

model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename=MODEL_FILENAME,
    token=os.getenv("HF_TOKEN") or None
)

print("Model file ready. Loading TensorFlow model...")

model = tf.keras.models.load_model(
    model_path,
    compile=False
)

print("Model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        f"Model has {model.output_shape[-1]} outputs, "
        f"but {len(class_names)} class names are defined. "
        "Verify the model's original class order."
    )


# --------------------------------------------------
# 3. FLASK SERVER
# --------------------------------------------------

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Tomato disease AI server is running.",
        "model": HF_REPO_ID,
        "status": "ready"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })


# --------------------------------------------------
# 4. IMAGE PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({
                "error": "No image received. Send the image "
                         "using the multipart field named 'image'."
            }), 400

        image_bytes = request.files["image"].read()

        if not image_bytes:
            return jsonify({"error": "Uploaded image is empty."}), 400

        # Decode the camera image into RGB.
        with Image.open(io.BytesIO(image_bytes)) as source:
            img = source.convert("RGB")
            img = img.resize(IMG_SIZE)

            # Match the preprocessing in your desktop script:
            # float32 RGB pixels, with no division by 255.
            img_array = np.asarray(img, dtype=np.float32)

        img_array = np.expand_dims(img_array, axis=0)

        raw_predictions = np.asarray(
            model.predict(img_array, verbose=0)[0],
            dtype=np.float64
        )

        if (
            raw_predictions.ndim != 1
            or raw_predictions.size != len(class_names)
            or not np.all(np.isfinite(raw_predictions))
        ):
            raise ValueError(
                "Model returned invalid predictions. "
                "Check the model output and class list."
            )

        # Preserve the existing score-conversion logic.
        if (
            np.any(raw_predictions < 0)
            or np.any(raw_predictions > 1)
            or not np.isclose(
                np.sum(raw_predictions), 1.0, atol=0.01
            )
        ):
            shifted = raw_predictions - np.max(raw_predictions)
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

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Image prediction failed. Check the server logs."
        }), 500


# --------------------------------------------------
# 5. LOCAL RUNNING
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
