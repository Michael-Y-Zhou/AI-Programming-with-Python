import argparse
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim


parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth',help = 'Path to the folder to save check point')
parser.add_argument('--arch', type = str, default = 'vgg19', help = 'Enter the model architecture to use:: "vgg11", "vgg13", "vgg16","vgg19","squeezenet1_0", "squeezenet1_1","alexnet", "densenet121", "densenet161", "densenet169", "densenet201","resnet18", "resnet34", "resnet50", "resnet101", "resnet152"')
parser.add_argument('--learning_rate', type = int, default = 0.003, help = 'The model learning rate')
parser.add_argument('--epochs', type = int, default = 10, help = 'Enter the epochs number')
parser.add_argument('--hidden_units', type = int, default = 512, help = 'Enter the number of hidden layers')
parser.add_argument('--gpu',action = 'store_true', help = 'Enable gpu by enter --gpu')
args = parser.parse_args()

def train_model(args):
    
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size=64)


    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if args.arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif args.arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif args.arch == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif args.arch == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif args.arch == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif args.arch == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif args.arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif args.arch == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif args.arch == 'resnet152':
        model = models.resnet152(pretrained=True)
    elif args.arch == 'squeezenet1_0':
        model = models.squeezenet1_0(pretrained=True)
    elif args.arch == 'squeezenet1_1':
        model = models.squeezenet1_1(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)


    for para in model.parameters():
        para.require_grad = False

    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features,args.hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(args.hidden_units,102),
                                    nn.LogSoftmax(dim = 1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
    model.to(device)


    # In[6]:

    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 25
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        acc = torch.mean(equals.type(torch.FloatTensor))
                        accuracy += acc.item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validationloader):.3f}.. "
                      f"Validation accuracy: {accuracy*100/len(validationloader):.3f}%")
                running_loss = 0
                model.train()

    
    
    checkpoint = {'hidden_layer':[each for each in model.classifier],
                          'model_state_dict':model.classifier.state_dict(),
                          'optimizer_state_dict':optimizer.state_dict(),
                          'epochs':epochs,
                          'class_to_idx':train_datasets.class_to_idx,
                          'model_architecture':args.arch
                         }
    file_path = args.save_dir
    torch.save(checkpoint,file_path)


train_model(args)
