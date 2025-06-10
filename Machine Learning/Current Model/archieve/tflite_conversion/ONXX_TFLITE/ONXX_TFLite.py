import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# ========================
# File Paths
# ========================
onnx_model_path = "resnet18_emotion.onnx"
tflite_model_path = "resnet18_emotion.tflite"

# ========================
# 1. Load the ONNX Model
# ========================
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

# ========================
# 2. Convert ONNX to TensorFlow Model
# ========================
print("Converting ONNX to TensorFlow...")

# Prepare the TensorFlow representation
tf_rep = prepare(onnx_model)
model = tf_rep.tf_module

# Export to SavedModel format
saved_model_dir = "tf_model"
tf_rep.export_graph(saved_model_dir)

# ========================
# 3. Convert TensorFlow to TFLite Model
# ========================
print("Converting TensorFlow to TFLite...")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {tflite_model_path}")
