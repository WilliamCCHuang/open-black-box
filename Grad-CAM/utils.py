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


def preprocess_image(pil_img):
    tensor = transforms.ToTensor()(pil_img)
    tensor = torch.unsqueeze(tensor, dim=0)
    
    return tensor


def load_class_idx():
    with open('../imagenet_class_index.json') as f:
        class_idx = json.load(f)
    
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    
    return idx2label


