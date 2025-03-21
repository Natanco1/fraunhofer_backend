import numpy as np
import tensorflow as tf
import os
import tqdm
from PIL import Image
import matplotlib.pyplot as plt

class StyleTransfer:
    def __init__(self, img_size=400, vgg_weights_path=None):
        self.img_size = img_size
        self.vgg = tf.keras.applications.VGG19(
            include_top=False,
            input_shape=(img_size, img_size, 3),
            weights=vgg_weights_path
        )
        self.vgg.trainable = False
        self.STYLE_LAYERS = [
            ('block1_conv1', 0.2),
            ('block2_conv1', 0.2),
            ('block3_conv1', 0.2),
            ('block4_conv1', 0.2),
            ('block5_conv1', 0.2)
        ]
        self.content_layer = [('block5_conv4', 1)]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=10e-3)  # match the notebook
        self.vgg_model_outputs = self.get_layer_outputs(self.vgg, self.STYLE_LAYERS + self.content_layer)

    def load_image(self, path):
        image = np.array(Image.open(path).resize((self.img_size, self.img_size)))
        return tf.constant(np.reshape(image, ((1,) + image.shape)))

    def compute_content_cost(self, content_output, generated_output):
        a_C = content_output[-1]
        a_G = generated_output[-1]
        _, n_H, n_W, n_C = a_G.shape.as_list()
        J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(a_C - a_G))
        return J_content

    def gram_matrix(self, A):
        return tf.matmul(A, A, transpose_b=True)

    def compute_layer_style_cost(self, a_S, a_G):
        _, n_H, n_W, n_C = a_G.shape
        a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))
        GS = self.gram_matrix(a_S)
        GG = self.gram_matrix(a_G)
        J_style_layer = (1 / (4 * n_C ** 2 * (n_H * n_W) ** 2)) * tf.reduce_sum(tf.square(GS - GG))
        return J_style_layer

    def compute_style_cost(self, style_image_output, generated_image_output):
        J_style = 0
        a_S = style_image_output[:-1]
        a_G = generated_image_output[:-1]
        for i, weight in enumerate(self.STYLE_LAYERS):  
            J_style += weight[1] * self.compute_layer_style_cost(a_S[i], a_G[i])
        return J_style

    @tf.function()
    def total_cost(self, J_content, J_style, alpha=10, beta=40):
        return alpha * J_content + beta * J_style

    def clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def tensor_to_image(self, tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            tensor = tensor[0]
        return Image.fromarray(tensor)

    def get_layer_outputs(self, vgg, layer_names):
        outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
        return tf.keras.Model([vgg.input], outputs)

    @tf.function()
    def train_step(self, generated_image, a_C, a_S):
        with tf.GradientTape() as tape:
            a_G = self.vgg_model_outputs(generated_image)
            J_style = self.compute_style_cost(a_S, a_G)
            J_content = self.compute_content_cost(a_C, a_G)
            J = self.total_cost(J_content, J_style)  
        
        grad = tape.gradient(J, generated_image)
        self.optimizer.apply_gradients([(grad, generated_image)])
        generated_image.assign(self.clip_0_1(generated_image))
        return J

    def train(self, content_path, style_path, output_dir="../media/output", epochs=2501):
        os.makedirs(output_dir, exist_ok=True)
        content_image = self.load_image(content_path)
        style_image = self.load_image(style_path)
        
        # Initial guess for the generated image: Start with content image
        generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        
        # OPTIONAL: Add small random noise (not too much, just a little to break symmetry)
        noise = tf.random.normal(tf.shape(generated_image), mean=0.0, stddev=0.05)
        generated_image.assign(generated_image + noise)
        
        a_C = self.vgg_model_outputs(content_image)
        a_S = self.vgg_model_outputs(style_image)
        
        for i in tqdm.tqdm(range(epochs), desc="Training Progress"):
            J = self.train_step(generated_image, a_C, a_S)
            
            # Print loss every 100 iterations
            if i % 100 == 0:
                print(f"Epoch {i} - Loss: {J}")
                image = self.tensor_to_image(generated_image)
                image.save(f"{output_dir}/image_{i}.jpg")
                plt.imshow(image)
                plt.show()
        
        return generated_image
