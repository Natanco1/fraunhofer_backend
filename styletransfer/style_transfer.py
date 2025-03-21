import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

class StyleTransfer:
    def __init__(self, content_weight=1e3, style_weight=1e-2, total_variation_weight=1e-6):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight
        self.vgg_model = self._load_vgg_model()

    def _load_vgg_model(self):
        """Load the VGG19 model for feature extraction."""
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg19.trainable = False
        return vgg19

    def _preprocess_image(self, image_path):
        """Preprocess an image to match the input requirements of VGG19."""
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return tf.convert_to_tensor(image, dtype=tf.float32)

    def _deprocess_image(self, image):
        """Deprocess the image back to a viewable format."""
        image = image.numpy().squeeze()
        image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)

    def _get_content_and_style_features(self, content_image, style_image):
        """Extract content and style features using the VGG19 model."""
        content_layers = ['block5_conv2']
        style_layers = [
            'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'
        ]

        model = tf.keras.models.Model(inputs=self.vgg_model.input, outputs=[self.vgg_model.get_layer(layer).output for layer in content_layers + style_layers])

        content_features = model(content_image)[0]
        style_features = model(style_image)
        return content_features, style_features

    def _compute_content_loss(self, content_image, target_image):
        """Compute the content loss."""
        return tf.reduce_mean(tf.square(content_image - target_image))

    def _compute_style_loss(self, style_features, target_features):
        """Compute the style loss."""
        style_loss = 0
        for style, target in zip(style_features, target_features):
            style_loss += tf.reduce_mean(tf.square(self._gram_matrix(style) - self._gram_matrix(target)))
        return style_loss

    def _compute_total_variation_loss(self, image):
        """Compute total variation loss for smoothness."""
        x_variation = image[:, 1:, :, :] - image[:, :-1, :, :]
        y_variation = image[:, :, 1:, :] - image[:, :, :-1, :]
        return tf.reduce_sum(tf.abs(x_variation)) + tf.reduce_sum(tf.abs(y_variation))

    def _gram_matrix(self, input_tensor):
        """Compute the Gram matrix for a given tensor."""
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        return result / tf.cast(input_tensor.shape[1] * input_tensor.shape[2], tf.float32)


    def _total_loss(self, content_features, style_features, target_content_features, target_style_features, target_image):
        """Compute the total loss (content + style + total variation)."""
        content_loss = self._compute_content_loss(content_features, target_content_features)
        style_loss = self._compute_style_loss(style_features, target_style_features)
        total_variation_loss = self._compute_total_variation_loss(target_image)
        
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss + self.total_variation_weight * total_variation_loss
        return total_loss


    def transfer_style(self, content_image_path, style_image_path, num_iterations=100, learning_rate=5.0):
        """Perform style transfer."""
        content_image = self._preprocess_image(content_image_path)
        style_image = self._preprocess_image(style_image_path)

        target_image = tf.Variable(content_image)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        with tqdm(range(num_iterations), desc="Style Transfer Progress", unit="iteration") as pbar:
            for iteration in pbar:
                with tf.GradientTape() as tape:
                    tape.watch(target_image)
                    
                    content_features, style_features = self._get_content_and_style_features(content_image, style_image)
                    target_content_features, target_style_features = self._get_content_and_style_features(target_image, style_image)

                    
                    loss = self._total_loss(content_features, style_features, target_content_features, target_style_features, target_image)
                
                gradients = tape.gradient(loss, target_image)
                gradients = tf.clip_by_value(gradients, -1.0, 1.0)
                optimizer.apply_gradients([(gradients, target_image)])

                pbar.set_postfix({"loss": loss.numpy()})

        return self._deprocess_image(target_image)
