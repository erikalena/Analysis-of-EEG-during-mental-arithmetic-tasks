import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import os
import datetime
from utils.utils import *
from utils.read_data import *
from classifier.models import *
from classifier.training import *

@dataclass
class Config:
    """
    A class to store all the configuration parameters
    """
    curr_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    number_of_subjects: int = 36
    first_subj: int = 1
    datset_size: int = 0
    batch_size: int = 32
    epochs: int = 100
    network_type: str = 'resnet18'
    classification: str = 'ms'
    pretrained: bool = False
    nclasses: int = 2
    nelectrodes: int = 20
    ndim: int = 2
    input_channels: int = 20
    input_data: int = None
    dataset: str = 'eeg_mat'
    dir_path: str = './' + dataset + '/results_classifier/' + network_type + "_" + curr_time
    checkpoint_path: str = None#'./' + dataset + 'results_classifier/resnet18_20230920-142905'
    optimizer: optim = optim.Adam
    learning_rate: float = 0.001
    loss_fn: nn = nn.CrossEntropyLoss 
    train_rate: float = 0.8
    valid_rate: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def save_config(self, file_path):
        """
        Write all the configuration parameters to file
        """
        with open(file_path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f'{key}: {value}\n')



CONFIG = Config()



def run(dataset):

    # make directory to save results
    if not os.path.isdir(CONFIG.dir_path):
        os.mkdir(CONFIG.dir_path)

    # save configuration parameters
    file_path = CONFIG.dir_path + '/config.txt'

    CONFIG.save_config(file_path)

    # build data loaders
    print("Building data loaders...", flush=True)

    if CONFIG.checkpoint_path is not None:
        # load dataloaders from file
        file_dataloaders = CONFIG.checkpoint_path + '/dataloaders.pkl'
        dataloaders = load_dataloaders(file_dataloaders)
        trainloader = dataloaders['train']
        validloader = dataloaders['val']
        testloader = dataloaders['test']
    else:
        trainloader, validloader, testloader = build_dataloader(dataset, batch_size=CONFIG.batch_size, train_rate=CONFIG.train_rate, valid_rate=CONFIG.valid_rate, shuffle=True) 
    
    dataloaders = {'train': trainloader, 'val': validloader, 'test': testloader}

    spectrogram, _, _, _, _ = next(iter(trainloader))

    print("Spectrogram shape: ", spectrogram.shape, flush=True)
    print("Min spectrogram: ", torch.min(spectrogram), flush=True)
    print("Max spectrogram: ", torch.max(spectrogram), flush=True)

    # save dataloaders to file
    file_dataloaders = CONFIG.dir_path + '/dataloaders.pkl'
    save_dataloaders(dataloaders, file_dataloaders)

    # load model
    if CONFIG.network_type.lower() == 'shallownet':
        print("Loading shallowNet...", flush=True)
        input_size = dataset.spectrograms[0].shape
        if len(dataset.spectrograms[0].shape) > 2:
            input_size = dataset.spectrograms[0][0].shape

        model = shallow2d(nclasses = CONFIG.nclasses, input_size = input_size, input_channels=CONFIG.input_channels, device = CONFIG.device)
    else:
        model = load_resnet18(nclasses = CONFIG.nclasses, pretrained = CONFIG.pretrained, device = CONFIG.device, input_channels=CONFIG.input_channels)

    # move model parameters to device
    model = model.to(CONFIG.device)
    model.device = CONFIG.device

    # define loss function and optimizer
    class_weights = get_weights(dataset, CONFIG.nclasses)
    loss_fn = CONFIG.loss_fn(weight=class_weights, reduction='mean')
    print("Weights: ", class_weights, flush=True)

    # move loss function to device
    loss_fn = loss_fn.to(CONFIG.device)

    # train model
    print("\n\nTraining model...", flush=True)
    file_checkpoint = CONFIG.checkpoint_path + '/checkpoint.pt' if CONFIG.checkpoint_path is not None else None
    load = True if CONFIG.checkpoint_path is not None and os.path.isfile(file_checkpoint) else False
    model = train_model(model, loss_fn, dataloaders, num_epochs=CONFIG.epochs, folder = CONFIG.dir_path, load_checkpoint = load, checkpoint_path = file_checkpoint, device=CONFIG.device)

    # test the model
    print("\n\nTesting model...", flush=True)
    test_acc, _, ytrue, ypred = test_model(model, testloader, folder = CONFIG.dir_path)
    #_ = get_confusion_matrix(ytrue, ypred, path = CONFIG.dir_path)



if __name__ == "__main__":
    
    # load data from folder
    sample_data_folder = './eeg_mat/eeg_data/'
    
    # check that configuration parameters are consistent
    assert CONFIG.network_type.lower() in ['shallownet', 'resnet18'], "network type must be either shallowNet or resnet18"
    if CONFIG.network_type == 'shallowNet':
        CONFIG.ndim == 1
        CONFIG.input_channels == 64
    else:
        CONFIG.ndim == 2 


    file_path = f'./saved_datasets/eeg_dataset_ns_{CONFIG.number_of_subjects}_ch_{CONFIG.input_channels}_nc_{CONFIG.nclasses}_{CONFIG.classification}_05sec_resample.pkl'

    # if dataset already exists, load it
    load = True if os.path.isfile(file_path) else False
    
    if load: 
        print("Loading dataset...", flush=True)
        dataset = load_dataset(file_path)
    else:
        print("Creating dataset...", flush=True)
        dataset = read_eeg_data(sample_data_folder, file_path, input_channels=CONFIG.input_channels, number_of_subjects=CONFIG.number_of_subjects, type = CONFIG.classification)
        
        
    CONFIG.dataset_size = len(dataset)
    print(dataset.spectrograms[0].shape, flush=True)

    run(dataset)
    
