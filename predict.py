# Dependencies
import json
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from get_input_args import get_input_args_predict

### Functions
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transformer(image)

def predict(image_path, model, topk=5, gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    model.to(device)
    image = process_image(image_path)
    image = image.unsqueeze_(0).float()
    
    with torch.no_grad():
        output = model.forward(image.cuda())
    p = F.softmax(output.data, dim=1)
    p = p.topk(topk)
    return p

### Loading model
model = torch.load('checkpoint.pt')
### Arg Parser
in_arg = get_input_args_predict()
### Prediction
p = predict(in_arg.input, model, in_arg.top_k, in_arg.gpu)
with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)
vals = np.array(p[0][0])
cats = [
    cat_to_name[str(i+1)]
    for i in np.array(p[1][0])
]
p_dic = {k:v for k, v in zip(cats, vals)}
print(p_dic)