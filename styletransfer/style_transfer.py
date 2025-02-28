import tensorflow as tf
import numpy as np
import tqdm
import pprint
import matplotlib.pyplot as plt
from PIL import Image

class StyleTransfer:
    def __init__(self, content_path, style_path, img_size=400, vgg_weights='/home/natanco-brain/projects/fraunhofer/fraunhofer_backend/styletransfer/models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'):
        self.img_size = img_size
        self.vgg = tf.keras.applications.VGG19(include_top=False,
                                               input_shape=(img_size, img_size, 3),
                                               weights=vgg_weights)
        self.vgg.trainable = False
        
        self.STYLE_LAYERS = [
            ('block1_conv1', 0.2),
            ('block2_conv1', 0.2),
            ('block3_conv1', 0.2),
            ('block4_conv1', 0.2),
            ('block5_conv1', 0.2)
        ]
        self.CONTENT_LAYER = [('block5_conv4', 1)]
        
        self.content_image = self.load_and_process_image(content_path)
        self.style_image = self.load_and_process_image(style_path)
        self.generated_image = tf.Variable(tf.image.convert_image_dtype(self.content_image, tf.float32))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)
        
        self.model = self.get_layer_outputs(self.vgg, self.STYLE_LAYERS + self.CONTENT_LAYER)
        self.a_C = self.model(self.content_image)
        self.a_S = self.model(self.style_image)

    def load_and_process_image(self, path):
        image = np.array(Image.open(path).resize((self.img_size, self.img_size)))
        return tf.constant(np.reshape(image, ((1,) + image.shape)))
    
    def get_layer_outputs(self, vgg, layer_names):
        outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
        return tf.keras.Model([vgg.input], outputs)
    
    def compute_content_cost(self, content_output, generated_output):
        a_C = content_output[-1]
        a_G = generated_output[-1]
        m, n_H, n_W, n_C = a_G.shape.as_list()
        a_C_unrolled = tf.reshape(a_C, shape=[m, -1, n_C])
        a_G_unrolled = tf.reshape(a_G, shape=[m, -1, n_C])
        return (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    
    def gram_matrix(self, A):
        return tf.matmul(A, A, transpose_b=True)
    
    def compute_layer_style_cost(self, a_S, a_G):
        m, n_H, n_W, n_C = a_G.shape
        a_S = tf.reshape(a_S, shape=[-1, n_C])
        a_G = tf.reshape(a_G, shape=[-1, n_C])
        GS = self.gram_matrix(a_S)
        GG = self.gram_matrix(a_G)
        return (1 / (4 * n_C **2 * (n_H * n_W) **2)) * tf.reduce_sum(tf.square(GS - GG))
    
    def compute_style_cost(self, style_image_output, generated_image_output):
        J_style = 0
        a_S = style_image_output[:-1]
        a_G = generated_image_output[:-1]
        for i, weight in zip(range(len(a_S)), self.STYLE_LAYERS):
            J_style += weight[1] * self.compute_layer_style_cost(a_S[i], a_G[i])
        return J_style
    
    @tf.function()
    def total_cost(self, J_content, J_style, alpha=10, beta=40):
        return alpha * J_content + beta * J_style
    
    @tf.function()
    def train_step(self):
        with tf.GradientTape() as tape:
            a_G = self.model(self.generated_image)
            J_content = self.compute_content_cost(self.a_C, a_G)
            J_style = self.compute_style_cost(self.a_S, a_G)
            J = self.total_cost(J_content, J_style)
        grad = tape.gradient(J, self.generated_image)
        self.optimizer.apply_gradients([(grad, self.generated_image)])
        self.generated_image.assign(self.clip_0_1(self.generated_image))
        return J
    
    def clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    
    def tensor_to_image(self, tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)
    
    def train(self, epochs=10):
        for i in tqdm.tqdm(range(epochs), desc="Training Progress"):
            self.train_step()
            if i % 250 == 0:
                print(f"Epoch {i}")
                image = self.tensor_to_image(self.generated_image)
                image.save(f"../media/output/image_{i}.jpg")
                plt.imshow(image)
                plt.show()
    
    def generate_image(self):
        return self.tensor_to_image(self.generated_image)