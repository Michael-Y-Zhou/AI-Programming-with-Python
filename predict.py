import argparse
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
import json
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('checkpoint')


parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Path to cat_to_name.json')
parser.add_argument('--top_k', type = int, default = 5, help = 'The number of the posible results')
parser.add_argument('--gpu',action = 'store_true', help = 'Enable gpu by enter --gpu')
args = parser.parse_args()

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(filepath):
    # Load the checkpoint
    
    checkpoint = torch.load(filepath)
    model_arch = checkpoint['model_architecture']
    # Renuild the model
    if model_arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif model_arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif model_arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model_arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_arch == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif model_arch == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif model_arch == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif model_arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_arch == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_arch == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif model_arch == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif model_arch == 'squeezenet1_0':
        model = models.squeezenet1_0(pretrained=True)
    elif model_arch == 'squeezenet1_1':
        model = models.squeezenet1_1(pretrained=True)

    model.classifier = nn.Sequential(*checkpoint['hidden_layer'])
    model.classifier.load_state_dict(checkpoint['model_state_dict'])

    # Rebuild the optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epochs = checkpoint['epochs']
    class_to_idx = checkpoint['class_to_idx']

    return model, optimizer, class_to_idx, epochs

model, optimizer, class_to_idx, epochs = load_checkpoint(args.checkpoint)

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image_path)

    # Resize the image to 256 regardless the width and height
    desired_size = 256
    old_size = im.size

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    # Crop the image to expected size(224,224)
    size_224 = (16,16,240,240)
    croped_im = new_im.crop(size_224)

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    nomal_image = (np.array(croped_im)/255 - mean)/std

    # Transpose the image shape and cast it to tensor
    return torch.from_numpy(nomal_image.transpose((2,1,0)))

def predict(image, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    dim_up = torch.tensor(np.array([image.numpy()])).type(torch.FloatTensor)

    with torch.no_grad():
        logps = model.forward(dim_up.to(device))
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim = 1)


        idx_to_class = {value: key for key,value in class_to_idx.items()} # Reverse the dictionary
        prob = [p.item() for p in top_p[0].data] # Cast tensor into a list
        classes = [idx_to_class[i.item()] for i in top_class[0].data]
        flower_classes = [cat_to_name[str(i)] for i in classes]

    return prob, flower_classes

if args.gpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
model.to(device)
file_path = args.input

processed_image = process_image(file_path)
flower_posibility, flower_classes = predict(processed_image, model, device, topk=args.top_k)
for i in range(len(flower_classes)):
    print(f'flower_classes: {flower_classes[i]}: {flower_posibility[i]*100:.2f}%')

