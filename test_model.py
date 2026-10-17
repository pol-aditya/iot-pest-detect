import tensorflow as tf

model = tf.keras.models.load_model(
    "my_model.keras",
    compile=False
)

model.summary()

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)