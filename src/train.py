import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torchvision import  transforms, datasets, models
from torch.utils.data import DataLoader, random_split

from helper import train_model_wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = 'dataset'

if __name__ == '__main__':
    #Parameter
    batchsize = 16
    epoch = 20

    transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor()
    ])

    #Load Dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    #Split Dataset
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, validation_set = random_split(dataset, [train_size, val_size])

    #Dataloader
    train_loader = DataLoader(train_set, batchsize, shuffle=True)
    val_loader = DataLoader(validation_set, batchsize, shuffle=False)

    #Model
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 9)
    model = model.to(device)

    #Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Train
    train_model_wandb(model,train_loader,val_loader,criterion,optimizer,20)