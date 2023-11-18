import time
import tensorflow as tf
from tensorflow.python.eager.context import PhysicalDevice


def test_tensorflow_is_working():
    assert tf.__version__ == "2.5.0"
    gpu = tf.config.list_physical_devices("GPU")
    assert gpu[0] == PhysicalDevice(name="/physical_device:GPU:0", device_type="GPU")


def test_gpu():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    )
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )
    assert train_dataset.element_spec == tf.TensorSpec(
        shape=(None, 28, 28, 1), dtype=tf.float32, name=None
    )
