import tensorflow as tf

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Verify basic operations
hello = tf.constant('Hello, TensorFlow!')
print(hello.numpy())
