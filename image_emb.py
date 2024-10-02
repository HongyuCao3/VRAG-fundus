import clip
import os
import torch
from PIL import Image
import numpy as np
class ImgEmb():
    def __init__(self, device):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device
    
    def get_features_from_image_path(self, image_paths):
        images = [self.preprocess(Image.open(image_path).convert("RGB")) for image_path in image_paths]
        image_input = torch.tensor(np.stack(images)).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
        return image_features
        
        
if __name__ == "__main__":
    device = "cuda:1"
    img_path = "./segmentation/"
    image_paths = [os.path.join(img_path,file) for file in os.listdir(img_path)]
    IE = ImgEmb(device)
    print(image_paths)
    feature = IE.get_features_from_image_path(image_paths)
    print(feature)