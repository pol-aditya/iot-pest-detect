import tensorflow as tf
import keras
from keras.layers import Dense

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

class CompatibleDense(Dense):
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return cls(**config)

# Register compatibility handler under the saved layer name
keras.utils.get_custom_objects()["Dense"] = CompatibleDense

print("\nLoading model...")

model = tf.keras.models.load_model(
    "best_EfficientNetB2 (1).keras",
    compile=False,
    safe_mode=True
)

print("\nSUCCESS: Model loaded!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
model.summary()
