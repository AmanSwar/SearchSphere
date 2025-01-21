
import mobileclip
import torch
from PIL import Image
import os
model , _ , preprocess = mobileclip.create_model_and_transforms('mobileclip_s0' , pretrained=r"/home/aman/weights/mobileclip_s0.pt")
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
def text_extract(text:str):
    """
    Function to generate text embeddings using mobileclip
    args:
        text:str = text content which needs to embedded

    return:
        text_features:torch.tensor = embeddings of text
    """
    global model , tokenizer
    inputs = tokenizer([text])
    with torch.no_grad():
        text_features = model.encode_text(inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features =  text_features.cpu().numpy()

    return text_features


def image_extract(image_path: os.path):
    """
    Function to generate image embeddings using mobileCLIP
    args:
        image_path:str = path of the image
    return:
        image_feature:torch.tensor = embedding of image
    """
    global model , preprocess

    image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        image_features = image_features.cpu().numpy()
    return image_features


