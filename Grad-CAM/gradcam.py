import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import preprocess_image, load_class_idx


class GradCAM:
    
    def __init__(self, extractor, transform=None):
        self.probas = None
        self.outputs = None
        self.gradients = None
        self.input_size = None
        self.feature_maps = None

        self.extractor = extractor
        self.transform = transform
        self.idx2label = load_class_idx()
    
    def _predict(self, img_tensor):
        with torch.no_grad():
            logits = self.extractor(img_tensor)
            probas = F.softmax(logits, dim=-1)

        self.probas = probas.detach().numpy().squeeze()  # (n_classes,)

    def _save_gradients(self, grad):
        self.gradients = grad.detach().numpy()[0]
    
    def _forward(self, img_tensor, use_logits):
        feature_maps = self.extractor.features(img_tensor)
        logits = self.extractor.classifier(feature_maps)
        probas = F.softmax(logits, dim=-1)
        
        feature_maps.register_hook(self._save_gradients)
        
        if use_logits:
            self.outputs = logits
        else:
            self.outputs = probas

        self.feature_maps = feature_maps.detach().numpy().squeeze()  # (c, h, w)
        
    def _backward(self, class_idx):    
        onehot_target = torch.zeros(self.probas.shape, dtype=torch.float)  # (n_classes,)
        onehot_target = torch.unsqueeze(onehot_target, dim=0)  # (1, n_classes)
        onehot_target[0][class_idx] = 1
        
        self.extractor.model.zero_grad()
        self.outputs.backward(gradient=onehot_target)
        
    def _get_class_idx(self, i):
        class_idx = self.probas.argsort()
        class_idx = class_idx[-i]
        
        return class_idx
    
    def generate_heatmap(self, img, class_idx, counterfactual=False, use_logits=True, relu_on_gradients=False):
        if isinstance(img, (Image.Image, np.ndarray)):
            if not self.transform:
                raise ValueError('Need to asign `transform` to preprocess `img`.')

            img_tensor = self.transform(img)
        elif isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            raise ValueError('Type of `img` should be `Image.Image`, `np.ndarray`, or `torch.Tensor`')
        
        self.input_size = list(img_tensor.size())[2:]

        self._forward(img_tensor, use_logits)
        self._backward(class_idx)
        
        gradients = self.gradients if not counterfactual else - self.gradients
        
        if relu_on_gradients:
            weights = np.mean(np.maximum(gradients, 0), axis=(1, 2))
        else:
            weights = np.mean(gradients, axis=(1, 2))
            
        weights = weights.reshape((-1, 1, 1))
        
        heatmap = np.sum(weights * self.feature_maps, axis=0)  # weighted sum over feature maps
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)  # Normalize between 0-1
        heatmap = np.uint8(heatmap * 255)
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize(self.input_size, Image.ANTIALIAS)
        heatmap = np.array(heatmap)
        
        return heatmap
        
    def plot_image_heatmap(self, img, top=1, counterfactual=False, use_logits=True, relu_on_gradients=False):
        if isinstance(img, (Image.Image, np.ndarray)):
            if not self.transform:
                raise ValueError('Need to asign `transform` to preprocess `img`.')

            img_tensor = self.transform(img)
        elif isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            raise ValueError('Type of `img` should be `Image.Image`, `np.ndarray`, or `torch.Tensor`')
        
        self.input_size = list(img_tensor.size())[2:]

        self._predict(img_tensor)
        
        cols = top + 1
        plt.figure(figsize=(4 * cols, 4))
        
        for i in range(cols):
            if i == 0:
                plt.subplot(1, cols, i+1)
                plt.imshow(img, alpha=1.0)
                plt.title('original image')
                plt.axis('off')
            else:
                class_idx = self._get_class_idx(i)
                label = self.idx2label[class_idx]
                proba = self.probas[class_idx]
                heatmap = self.generate_heatmap(img_tensor, class_idx, counterfactual, relu_on_gradients)
                
                plt.subplot(1, cols, i+1)
                plt.imshow(img, alpha=1.0)
                plt.imshow(heatmap, cmap='rainbow', alpha=0.7)
                plt.title('{} ({:.3f})'.format(label, proba))
                plt.axis('off')
                