from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import time
from classifier.training import save_checkpoint



def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=16):
        UF=  input.view(input.size(0), size, 24)
        return UF

class EEG_CNN_VAE(nn.Module):
    def __init__(self, data_shape, in_channels=20, latent_dim=16, device = 'cpu'):
        super (EEG_CNN_VAE,self).__init__()

        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=10, stride = 2), 
            nn.BatchNorm1d(num_features=16), 
            nn.PReLU(), 
            nn.MaxPool1d(2),             
            Flatten()
        )
        
        eeg_size = self.compute_conv_output_size(data_shape)

        self.fc1 = nn.Linear(eeg_size, latent_dim) # one for mu and other for sigma
        self.fc2 = nn.Linear(eeg_size, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 384)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=22, stride=2),  
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=18, stride=2), 
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=in_channels, kernel_size=21, stride=4),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, sigma):

        std = sigma.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, sigma = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, sigma)
        return z, mu, sigma

    def encode(self, x):
        h = self.encoder(x)
        z, mu, sigma = self.bottleneck(h)
        return z, mu, sigma
    
    def decode(self, z):
        z = self.fc3(z)
        x = self.decoder(z)
        return x

    def forward(self, x):
        z, mu, sigma = self.encode(x)
        x = self.decode(z)
        return x, mu, sigma
    
    def compute_conv_output_size(self, data_shape):
        x = torch.rand(data_shape)
        output_feat = self.encoder(x)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size
    

def loss_fn(recon_x, x, mu, sigma):
    # mean square error
    mse = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    # sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = - torch.sum(1 + sigma - mu.pow(2) - sigma.exp())

    return mse + kld, mse, kld




# implement early stopping
class EarlyStopping:
    def __init__(self, patience=1):
        self.patience = patience
        self.min_delta = 0.0001
        self.counter = 0
        self.min_loss = np.inf
        self.warmup_epochs = 20

    def early_stop(self, validation_loss, epoch):
        if validation_loss < self.min_loss:
            self.min_loss = validation_loss
            self.counter = 0
            # correct min delta to be 10% of the min loss
            self.min_delta = self.min_loss * 0.1

        elif validation_loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience and epoch > self.warmup_epochs:
                return True
        return False


def train_vae(dataloaders, data_shape, latent_dim, load = False, model_path=None, device = 'cpu', num_epochs = 1000, folder = None):

    model = EEG_CNN_VAE(data_shape, latent_dim).to(device)
    model.apply(weights_init)
    model.device = device
    model.latent_dim = latent_dim

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    early_stopper = EarlyStopping(patience=300)

    if load == True and model_path != None:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # save on file checkpoint file name from which we are loading
        # and the epoch from which we are starting
        with open(os.path.join(folder, 'results.txt'), 'a') as f:
            f.write(f'Loading checkpoint {model_path} at epoch {start_epoch - 1}\n')

    # Create a temporary directory to save training checkpoints
    best_model = os.path.join(folder, 'best_model_params.pt')
    torch.save(model.state_dict(), best_model)    

    best_loss = 1000000

    with open(os.path.join(folder, 'results.txt'), 'a') as f:
        # print header
        f.write(f'epoch,train_loss,mse,kdl,val_loss,mse,kdl\n')

    since = time.time()

    for epoch in tqdm(range(num_epochs)):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            for _, raw, labels, _, _ in dataloaders[phase]:
                raw, labels = raw.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    recon_data, mu, sigma = model(raw)
                    loss, mse, kld = loss_fn(recon_data, raw, mu, sigma)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            # save checkpoint
            save_checkpoint(model, optimizer, os.path.join(folder, 'checkpoint.pt'), epoch)
            
            # print results to a file
            with open(os.path.join(folder, 'results.txt'), 'a') as f:
                if phase == 'train':
                    f.write(f'{epoch},{loss.item()},{mse.item()},{kld.item()},')
                else:
                    f.write(f'{loss.item()},{mse.item()},{kld.item()}\n')

            
            """ if epoch % 100 == 0:
                to_print = "Epoch[{}/{}] {} Loss: {:.6f} {:.6f} {:.6f}".format(epoch+1, num_epochs, phase, loss.item(), mse.item(), kld.item())
                print(to_print) """

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = copy.deepcopy(model)
            # save model
            torch.save(best_model.state_dict(), os.path.join(folder, 'best_model_params.pt'))

        # check if we need to early stop
        if early_stopper.early_stop(loss.item(), epoch):
            print('Early stopping\n')
            break
    
    time_elapsed = time.time() - since
    with open(os.path.join(folder, 'results.txt'), 'a') as f:
        f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')



    return model


def generate_data(model, num_data_to_generate, device = 'cpu'):
    new_data = []

    with torch.no_grad():
        model.eval()
        for _ in range(num_data_to_generate):
            z = torch.randn(1, model.latent_dim).to(device) # random from normal distribution with mean 0 and variance 1
            recon_data = model.decode(z).cpu().numpy()

            new_data.append(recon_data)

        new_data = np.concatenate(new_data) 
        new_data = np.asarray(new_data)

    return new_data
    
