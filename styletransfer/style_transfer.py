import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import tqdm


class StyleTransfer:
    def __init__(self, content_weight=1, style_weight=1000000, num_steps=5):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.num_steps = num_steps

        self.model = models.vgg19(pretrained=True).features.eval()

    def preprocess_image(self, image_path):
        """
        Preprocess the image by resizing and normalizing it
        """
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        return image

    def get_features(self, image):
        """
        Extract features from specific layers of VGG19
        """
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '28': 'conv5_1'
        }

        features = {}
        x = image
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def run_style_transfer(self, content_img, style_img):
        """
        Perform style transfer using content and style images
        """
        content_features = self.get_features(content_img)
        style_features = self.get_features(style_img)

        target = content_img.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([target])

        for step in tqdm.tqdm(range(self.num_steps), desc="Running Style Transfer", ncols=100):
            def closure():
                target.data.clamp_(0, 1)
                optimizer.zero_grad()

                target_features = self.get_features(target)

                content_loss = self.content_weight * torch.nn.functional.mse_loss(target_features['conv4_1'], content_features['conv4_1'])

                style_loss = 0
                for layer in style_features:
                    target_feature = target_features[layer]
                    style_feature = style_features[layer]
                    _, c, h, w = target_feature.size()
                    target_gram = target_feature.view(c, h * w).mm(target_feature.view(c, h * w).t())
                    style_gram = style_feature.view(c, h * w).mm(style_feature.view(c, h * w).t())
                    style_loss += torch.nn.functional.mse_loss(target_gram, style_gram) / (c * h * w)

                loss = content_loss + self.style_weight * style_loss
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)

        return target


    def save_image(self, tensor, filename="styled_image.jpg"):
        """
        Save the tensor as an image to disk
        """
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()
        image = unloader(image.squeeze(0))
        image.save(filename)
