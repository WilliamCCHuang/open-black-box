import json
from PIL import Image

import torch
from torchvision import transforms


def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert(mode='RGB')
    img = img.resize((299, 299))
    
    return img
    

def preprocess_image(img, mean, std):
    img_tensor = transforms.ToTensor()(img)
    img_tensor = transforms.Normalize(mean, std)(img_tensor)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)  # (1, 3, 299, 299)
    
    return img_tensor
    

def load_class_idx():
    with open('../imagenet_class_index.json') as f:
        class_idx = json.load(f)
    
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    
    return idx2label