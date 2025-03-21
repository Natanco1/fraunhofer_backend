import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class StyleTransfer:
    def __init__(self, content_image_path, style_image_path, max_res=512):
        self.content_image_path = content_image_path
        self.style_image_path = style_image_path
        self.max_res = max_res
        self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    def load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        long_side = max(shape)
        scaling_factor = self.max_res / long_side

        new_shape = tf.cast(shape * scaling_factor, tf.int32)
        image = tf.image.resize(image, new_shape)
        image = image[tf.newaxis, :]

        return image

    def tensor_to_image(self, tensor):
        tensor = np.array(tensor * 255, dtype=np.uint8)

        if np.ndim(tensor) > 3:
            tensor = tensor[0]

        return PIL.Image.fromarray(tensor)

    def transfer_style(self):
        content_image = self.load_image(self.content_image_path)
        style_image = self.load_image(self.style_image_path)

        stylized_image = self.model(content_image, style_image)[0]

        return self.tensor_to_image(stylized_image)
