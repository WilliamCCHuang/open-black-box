import json
from PIL import Image

import torch
import torchvision.transforms as transforms


def load_image(img_path, size=None):
    img = Image.open(img_path)
    img = img.convert(mode='RGB')
    
    if size is not None:
        img = img.resize(size)
    
    return img


def preprocess_image(img, mean, std):
    img_tensor = transforms.ToTensor()(img)
    img_tensor = transforms.Normalize(mean, std)(img_tensor)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    
    return img_tensor


def load_class_idx():
    with open('../imagenet_class_index.json') as f:
        class_idx = json.load(f)
    
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    
    return idx2label


