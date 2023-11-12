import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import classifier




class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.warmup_epochs = 20

    def early_stop(self, validation_loss, epoch):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience and epoch > self.warmup_epochs:
                return True
        return False


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)





def train_model(model, criterion, dataloaders, num_epochs=25, folder = None, load_checkpoint = False, checkpoint_path = None, device="cpu"):
    
    #  make directory to save results
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
    since = time.time()

    start_epoch = 0

    # load checkpoint if needed
    if load_checkpoint:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # save on file checkpoint file name from which we are loading
        # and the epoch from which we are starting
        with open(os.path.join(folder, 'results.txt'), 'a') as f:
            f.write(f'Loading checkpoint {checkpoint_path} at epoch {start_epoch - 1}\n')
 
    # move model parameters to device
    model = model.to(device)
    model.device = device

    print("Running on ", device, flush=True)
    

    # Create a temporary directory to save training checkpoints
    best_model = os.path.join(folder, 'best_model_params.pt')

    torch.save(model.state_dict(), best_model)
    best_acc = 0.0


    print(f'Training on {len(dataloaders["train"])} samples and validating on {len(dataloaders["val"])} samples', flush=True)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    early_stopper = EarlyStopper(patience=10, min_delta=0.001)

    with open(os.path.join(folder, 'results.txt'), 'a') as f:
        # print header
        f.write(f'epoch,train_loss,train_acc,val_loss,val_acc\n')

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}\n')

        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0
            size = 0
            for inputs, _, labels, _, _ in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                #labels = labels % 2 # change labels to 0 and 1
                size += inputs.shape[0]
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # if input channels is 64 change shape of input
                    if model.input_channels > 1 and len(inputs.shape) > 4:
                        inputs = inputs.squeeze(1)
                    
                    
                    if len(inputs.shape) <= 3:
                        inputs = inputs.unsqueeze(1)
                                            
                
                    outputs = model(inputs)
                   
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) # by default loss is averaged over batch size
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / size
            epoch_acc = running_corrects.double() / size
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model)

            # save checkpoint
            save_checkpoint(model, optimizer, os.path.join(folder, 'checkpoint.pt'), epoch)
            
            # print results to a file
            with open(os.path.join(folder, 'results.txt'), 'a') as f:
                if phase == 'train':
                    f.write(f'{epoch},{epoch_loss},{epoch_acc:.4f},')
                else:
                    f.write(f'{epoch_loss},{epoch_acc:.4f}\n')

        # check if we need to early stop
        if early_stopper.early_stop(epoch_loss, epoch):
            with open(os.path.join(folder, 'results.txt'), 'a') as f:
                f.write('Early stopping\n')
            break

    time_elapsed = time.time() - since
    with open(os.path.join(folder, 'results.txt'), 'a') as f:
        f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model))

    return model


def test_model(model, dataloader, folder = None):
    """
    Test the model on the test set
    """
    # get the device of the model
    device = model.device

    model.eval()   # Set model to evaluate mode
    running_corrects = 0

    # keep track of correclty classified instances
    correct_instances = []
    ytrue, ypred = [], []

    size = 0 # keep track of the size of the test set
    
    for inputs, _, labels, ids, channels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        size += inputs.shape[0]
        # forward
        with torch.no_grad():
            if model.input_channels > 1 and len(inputs.shape) > 4:
                inputs = inputs.squeeze(1)
            
            if len(inputs.shape) <= 3:
                inputs = inputs.unsqueeze(1)
     
                
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
        # get indices of correctly classified instances
        correct_idx = torch.where(preds == labels.data)[0]

        ytrue.extend(labels.data.cpu().numpy())
        ypred.extend(preds.cpu().numpy())

        # save correctly classified instances
        for i in correct_idx:
            correct_instances.append({'spectrogram': inputs[i][0].cpu().numpy(), 'label': labels[i],  'channel': None})
    
    test_acc = running_corrects.double() / size

    # print results to a file
    if folder is not None:
        with open(os.path.join(folder, 'results.txt'), 'a') as f:
            f.write(f'Test Acc: {test_acc:.4f}\n')
    else:
        print(f'Test Acc: {test_acc:.4f}', flush = True)
        
    return test_acc, correct_instances, ytrue, ypred
