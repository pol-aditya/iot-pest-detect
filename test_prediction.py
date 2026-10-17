
import numpy as np
import keras
from PIL import Image

MODEL_PATH = "best_EfficientNetB2 (1).keras"
IMAGE_PATH = "bad.jpeg"  # Change this to your image filename

# Load the trained model
model = keras.models.load_model(MODEL_PATH, compile=False)

# Load and resize the test image
img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize((260, 260))

# Convert image to a batch
img_array = np.array(img, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
predictions = model.predict(img_array, verbose=0)
scores = predictions[0]

predicted_class = int(np.argmax(scores))

print("Predicted class index:", predicted_class)
print("All output scores:", scores)
print("Highest score:", float(np.max(scores)))
