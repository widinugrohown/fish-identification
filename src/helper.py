from matplotlib import pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function for training
def train(dataloader, model, loss_fn, optimizer, epoch):
    
    size = len(dataloader.dataset) # total number of images inside of loader
    num_batches = len(dataloader) # number of batches
    
    model.train()

    train_loss, correct = 0, 0
    train_accuracies = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update training loss
        train_loss += loss.item() # item() method extracts the loss’s value as a Python float

        # Calculate training accuracy
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    # Calculate average training loss and accuracy
    train_loss = train_loss / num_batches
    accuracy = 100 * correct / size

    train_accuracies.append(accuracy)

    # Print training accuracy and loss at the end of epoch
    print(f" Training Accuracy: {accuracy:.2f}%, Training Loss: {train_loss:.4f}")

def validation(dataloader, model, loss_fn,t):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    validation_loss, correct = 0, 0

    model.eval()
    validation_accuracies = []
    with torch.no_grad(): #  disable gradient calculation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            validation_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    

    validation_loss /= num_batches
    accuracy = 100 * correct / size

    validation_accuracies.append(accuracy)

    # Print test accuracy and loss at the end of epoch
    print(f" Validation Accuracy: {accuracy:.2f}%, Validation Loss: {validation_loss:.4f}")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import wandb

def train_model_wandb(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # Initialize wandb
    wandb.init(project="fish_identification", name="training")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        # Training Phase
        model.train()
        running_train_loss = 0.0
        correct_train_preds = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            correct_train_preds += torch.sum(preds == labels.data)
        
        train_loss = running_train_loss / len(train_loader.dataset)
        train_acc = correct_train_preds.double() / len(train_loader.dataset)
        
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val_preds = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                correct_val_preds += torch.sum(preds == labels.data)
        
        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = correct_val_preds.double() / len(val_loader.dataset)
        
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        print()
        
        # Log metrics to wandb
        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        
    torch.save(model.state_dict(),'model.pth')
    # Finish wandb run
    wandb.finish()
    

