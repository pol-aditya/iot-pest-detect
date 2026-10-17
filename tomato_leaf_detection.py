
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_EfficientNetB2 (1).keras"
)

RESULT_PATH = os.path.join(
    BASE_DIR,
    "prediction_result.txt"
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


# ============================================================
# 2. HELPER FUNCTION: FORMAT CLASS NAMES
# ============================================================

def format_class_name(class_name):
    name = class_name.replace("Tomato___", "")
    name = name.replace("_", " ")

    if "Spider mites" in name:
        return "Spider Mites / Two-Spotted Spider Mite"

    if name.lower() == "healthy":
        return "Healthy"

    return name


# ============================================================
# 3. LOAD THE TRAINED MODEL
# ============================================================

print("=" * 65)
print("       TOMATO LEAF DISEASE DETECTION SYSTEM")
print("=" * 65)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}\n\n"
        "Place 'best_EfficientNetB2 (1).keras' "
        "in the same folder as this Python script."
    )

print(f"\nModel file: {MODEL_PATH}")
print(
    f"Model size: "
    f"{os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB"
)
print("\nLoading model...")

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        f"The model returns {model.output_shape[-1]} outputs, "
        f"but {len(class_names)} class names are configured. "
        "Check the class order used during training."
    )


# ============================================================
# 4. GRAPHICAL IMAGE SELECTION WINDOW
# ============================================================

selected_image_path = ""

root = tk.Tk()
root.title("Tomato Leaf Disease Detection")
root.geometry("750x650")
root.configure(bg="#edf5ed")
root.resizable(True, True)

title_label = tk.Label(
    root,
    text="Tomato Leaf Disease Detection",
    font=("Arial", 20, "bold"),
    bg="#edf5ed",
    fg="#216b36"
)
title_label.pack(pady=(18, 8))

subtitle_label = tk.Label(
    root,
    text="Select a tomato leaf image to analyse",
    font=("Arial", 11),
    bg="#edf5ed",
    fg="#333333"
)
subtitle_label.pack(pady=5)

preview_label = tk.Label(
    root,
    text="Your selected image will appear here",
    font=("Arial", 12),
    bg="white",
    fg="#666666",
    width=48,
    height=15,
    relief="groove",
    borderwidth=2
)
preview_label.pack(pady=15, padx=20)

path_label = tk.Label(
    root,
    text="No image selected",
    font=("Arial", 9),
    bg="#edf5ed",
    fg="#333333",
    wraplength=680
)
path_label.pack(pady=5)


def choose_image():
    global selected_image_path

    file_path = filedialog.askopenfilename(
        parent=root,
        title="Choose a tomato leaf image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    try:
        with Image.open(file_path) as original:
            preview = original.convert("RGB")
            preview.thumbnail((450, 330))

        photo = ImageTk.PhotoImage(preview)

        preview_label.config(
            image=photo,
            text="",
            width=0,
            height=0
        )

        # Keep a reference so Tkinter does not remove the image.
        preview_label.image = photo

        selected_image_path = file_path
        path_label.config(text=f"Selected: {file_path}")

    except Exception as error:
        messagebox.showerror(
            "Image Error",
            f"Could not open the selected image.\n\n{error}",
            parent=root
        )


def start_detection():
    if not selected_image_path:
        messagebox.showwarning(
            "No Image Selected",
            "Please click 'Choose Leaf Image' first.",
            parent=root
        )
        return

    root.destroy()


def cancel_detection():
    root.destroy()


button_frame = tk.Frame(root, bg="#edf5ed")
button_frame.pack(pady=12)

choose_button = tk.Button(
    button_frame,
    text="Choose Leaf Image",
    command=choose_image,
    font=("Arial", 12, "bold"),
    bg="#2e8b57",
    fg="white",
    activebackground="#246b45",
    activeforeground="white",
    padx=18,
    pady=10,
    cursor="hand2"
)
choose_button.pack(side=tk.LEFT, padx=8)

detect_button = tk.Button(
    button_frame,
    text="Detect Disease",
    command=start_detection,
    font=("Arial", 12, "bold"),
    bg="#1e6091",
    fg="white",
    activebackground="#164a70",
    activeforeground="white",
    padx=18,
    pady=10,
    cursor="hand2"
)
detect_button.pack(side=tk.LEFT, padx=8)

cancel_button = tk.Button(
    root,
    text="Exit",
    command=cancel_detection,
    font=("Arial", 10),
    bg="#777777",
    fg="white",
    padx=15,
    pady=5,
    cursor="hand2"
)
cancel_button.pack(pady=5)

root.mainloop()

if not selected_image_path:
    print("\nNo image selected. Program stopped.")
    raise SystemExit(0)

image_path = selected_image_path

print("\nSelected image:", image_path)


# ============================================================
# 5. LOAD AND PREPARE THE IMAGE
# ============================================================

try:
    img = load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

except Exception as error:
    raise RuntimeError(
        f"Could not prepare the selected image: {error}"
    ) from error

print("Image input shape:", img_array.shape)
print("\nAnalysing image...")


# ============================================================
# 6. PREDICT THE CLASS
# ============================================================

raw_predictions = model.predict(
    img_array,
    verbose=0
)[0]

# Convert to probabilities if the model output is not
# already a probability distribution.
if (
    np.any(raw_predictions < 0)
    or np.any(raw_predictions > 1)
    or not np.isclose(np.sum(raw_predictions), 1.0, atol=0.01)
):
    shifted = raw_predictions - np.max(raw_predictions)
    exp_values = np.exp(shifted)
    predictions = exp_values / np.sum(exp_values)
else:
    predictions = raw_predictions

predicted_index = int(np.argmax(predictions))
predicted_class = class_names[predicted_index]
disease_name = format_class_name(predicted_class)
confidence = float(predictions[predicted_index]) * 100


# ============================================================
# 7. DETERMINE HEALTH STATUS
# ============================================================

if "healthy" in predicted_class.lower():
    status = "HEALTHY"
    pesticide_decision = "NOT INDICATED BY MODEL"
else:
    status = "POSSIBLE DISEASE / PEST DETECTED"
    pesticide_decision = (
        "INSPECT LEAF; CONFIRM BEFORE TREATMENT"
    )


# ============================================================
# 8. DISPLAY THE IMAGE AND PREDICTION
# ============================================================

plt.figure(figsize=(8, 7))
plt.imshow(img)
plt.axis("off")

plt.title(
    f"Tomato Leaf Analysis\n"
    f"Prediction: {disease_name}\n"
    f"Model score: {confidence:.2f}%",
    fontsize=13
)

plt.tight_layout()
plt.show()


# ============================================================
# 9. PRINT THE FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("                 TOMATO LEAF AI ANALYSIS")
print("=" * 70)

print(f"Image                  : {os.path.basename(image_path)}")
print(f"Predicted Class        : {predicted_class}")
print(f"Disease / Condition    : {disease_name}")
print(f"Model Score            : {confidence:.2f}%")
print(f"Status                 : {status}")
print(f"Pesticide Intervention : {pesticide_decision}")

print("=" * 70)


# ============================================================
# 10. DISPLAY THE TOP 3 PREDICTIONS
# ============================================================

top_indices = np.argsort(predictions)[::-1][:3]

print("\nTOP 3 PREDICTIONS")
print("-" * 70)

for rank, idx in enumerate(top_indices, start=1):
    readable_name = format_class_name(class_names[idx])
    score = float(predictions[idx]) * 100

    print(f"{rank}. {readable_name:<45} {score:.2f}%")

print("-" * 70)


# ============================================================
# 11. SAVE THE RESULTS TO A TEXT FILE
# ============================================================

with open(RESULT_PATH, "w", encoding="utf-8") as file:
    file.write("TOMATO LEAF AI ANALYSIS\n")
    file.write("=" * 60 + "\n")
    file.write(f"Image: {os.path.basename(image_path)}\n")
    file.write(f"Predicted Class: {predicted_class}\n")
    file.write(f"Disease / Condition: {disease_name}\n")
    file.write(f"Model Score: {confidence:.2f}%\n")
    file.write(f"Status: {status}\n")
    file.write(
        f"Pesticide Intervention: {pesticide_decision}\n\n"
    )

    file.write("TOP 3 PREDICTIONS\n")
    file.write("-" * 60 + "\n")

    for rank, idx in enumerate(top_indices, start=1):
        readable_name = format_class_name(class_names[idx])
        score = float(predictions[idx]) * 100

        file.write(
            f"{rank}. {readable_name}: {score:.2f}%\n"
        )

print("\nInference completed successfully!")
print("Results saved to:", RESULT_PATH)
print(
    "\nNote: Model scores are not guaranteed probabilities "
    "of a real-world diagnosis. Confirm suspected diseases "
    "before applying any pesticide."
)
