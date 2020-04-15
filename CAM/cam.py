import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn.functional as F

from utils import preprocess_image, load_class_idx


class CAM:
    def __init__(self, model):
        self.model = model
        self.probas = None
        self.features = None
        self.size = model.input_size[1:]
        self.idx2label = load_class_idx()
        
        self.model.eval()
        
        weight, bias = list(self.model.last_linear.parameters())
        self.weight = weight.detach().numpy()  # (1000, 2048)
        
        del weight, bias
    
    def _forward(self, img):
        with torch.no_grad():
            features = self.model.features(img)  # (1, c, h, w)
            logits = self.model.logits(features)  # (1, n_classes)
            probas = F.softmax(logits, dim=-1)
            
        self.features = features.numpy().squeeze()  # (c, h, w)
        self.probas = probas.numpy().squeeze()  # (n_classes)
    
    def _get_class_idx(self, i):
        class_idx = self.probas.argsort()
        class_idx = class_idx[-i]
        
        return class_idx
    
    def _generate_heatmap(self, class_idx):
        weight_c = self.weight[class_idx, :]  # (c,)
        weight_c = weight_c.reshape((-1, 1, 1))  # (c, 1, 1)

        heatmap = np.sum(weight_c * self.features, axis=0)  # (h, w)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)  # Normalize between 0-1
        heatmap = np.uint8(heatmap * 255)
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize(self.size, Image.ANTIALIAS)
        heatmap = np.array(heatmap)
        
        return heatmap
    
    def plot_image_heatmap(self, img, top=1):
        img_tensor = preprocess_image(img, self.model.mean, self.model.std)  # (1, 3, 299, 299)
        
        self._forward(img_tensor)
        
        cols = top + 1
        plt.figure(figsize=(4 * cols, 4))
        
        for i in range(cols):
            if i == 0:
                plt.subplot(1, cols, i+1)
                plt.imshow(img, alpha=1.0)
                plt.title('Original image')
                plt.axis('off')
            else:
                class_idx = self._get_class_idx(i)
                label = self.idx2label[class_idx]
                proba = self.probas[class_idx]
                heatmap = self._generate_heatmap(class_idx)
                
                plt.subplot(1, cols, i+1)
                plt.imshow(img, alpha=1.0)
                plt.imshow(heatmap, cmap='jet', alpha=0.5)
                plt.title('{} ({:.3f})'.format(label, proba))
                plt.axis('off')
                