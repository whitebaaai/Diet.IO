import numpy as np
from PIL import Image
import cv2

def canny_selfie(image):
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    edges = cv2.Canny(image_array, 100, 200)
    return edges

import sys
sys.path.append("./clipseg/models")
from clipseg import CLIPDensePredT
from torchvision import transforms
import torch

def mask_image(image):
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load('./weights/rd64-uni.pth', 
                                     map_location=torch.device('cuda')), 
                                     strict=False)
    input_image = image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((512, 512)),
    ])
    img = transform(input_image).unsqueeze(0)
    
    prompts = ['area outside of face']
    with torch.no_grad():
        preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]
    
    preds = torch.sigmoid(preds[0][0])
    array = preds.numpy()

    # Scale the values in the array to the range 0-255
    array = (array * 255).astype(np.uint8)
    array = np.squeeze(array)

    # Scale the values in the array to the range 0-255
    preds = Image.fromarray(array)
    return preds

def main():
    test_image = Image.open("./selfie_input.jpg")
    mask = mask_image(test_image)
    mask.save("mask.png")

if __name__ == "__main__":
    main()
