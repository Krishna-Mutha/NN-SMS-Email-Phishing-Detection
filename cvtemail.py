import tensorflow as tf

# Path to SavedModel folder
saved_model_dir = "email_model"

# Create converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Required for Conv1D / Embedding models
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Optimize size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save file
with open("email_model_select.tflite", "wb") as f:
    f.write(tflite_model)

print("Email model conversion successful")
interpreter = tf.lite.Interpreter(model_path="email_model_select.tflite")
interpreter.allocate_tensors()

print(interpreter.get_input_details())
print(interpreter.get_output_details())