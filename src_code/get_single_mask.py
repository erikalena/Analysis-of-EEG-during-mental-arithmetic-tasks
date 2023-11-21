import torch
from dataclasses import dataclass
import os
import datetime
from utils.utils import *
from utils.read_data import *
from utils.mask_training import *

from classifier.models import *
from classifier.training import *




@dataclass
class Config:
    """
    A class to store all the configuration parameters
    """
    curr_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    number_of_subjects: int = 5
    datset_size: int = 0
    batch_size: int = 32
    start_idx: int = 60
    end_idx: int = 0
    nclasses: int = 2
    classification: str = 'cq'
    model_path: str = './results_classifier/resnet18_cq_36/best_model_params.pt' 
    save_figures: bool = True
    input_channels: int = 20
    train_rate: float = 0.8
    valid_rate: float = 0.1
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save_config(self):
        """
        Write all the configuration parameters to file
        """
        print("\nConfig:\n", flush=True)
        for key, value in self.__dict__.items():
            print(f'{key}: {value}', flush=True)



CONFIG = Config()


def load_classifier(dataset):
    print("Loading model...\n", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = load_resnet18(nclasses = CONFIG.nclasses, pretrained = False, device = device, input_channels=CONFIG.input_channels)

    # load model parameters
    model.load_state_dict(torch.load(CONFIG.model_path, map_location=torch.device('cpu')))
    # move model and model parameters to device
    model.to(device)

    _, _, testloader = build_dataloader(dataset, batch_size=CONFIG.batch_size, train_rate=CONFIG.train_rate, valid_rate=CONFIG.valid_rate, shuffle=True, resample=False)
    test_acc, _, _, _ = test_model(model, testloader, folder=None)

    print("Test accuracy: ", test_acc, flush=True)

    return model
    

if __name__ == "__main__":
    
    # load data from folder
    sample_data_folder = './eeg_data/'

    file_path = f'./saved_datasets/eeg_dataset_ns_{CONFIG.number_of_subjects}_ch_{CONFIG.input_channels}_nc_{CONFIG.nclasses}_{CONFIG.classification}_1sec.pkl'
    print(file_path)
    # save config parameters
    CONFIG.save_config()

    # if dataset already exists, load it
    load = True if os.path.isfile(file_path) else False
    
    if load: 
        print("Loading dataset...", flush=True)
        dataset = load_dataset(file_path)
    else:
        print("Creating dataset...", flush=True)
        dataset = read_eeg_data(sample_data_folder, file_path, input_channels=CONFIG.input_channels, number_of_subjects=CONFIG.number_of_subjects, type = CONFIG.classification)
        
        
    print("Size of data set: ", len(dataset))

    # load classfier
    model = load_classifier(dataset)
    

    lam = 0.01
    mask_path = f'./results_single_class_mask_{str(CONFIG.classification)}/' 
    if not os.path.isdir(mask_path):
        os.mkdir(mask_path)

    for i in range(CONFIG.nclasses):
        print("\n\nTraining mask for class: ", i, flush=True)

        mask_path_class = mask_path + f'class_{i}/'
        if not os.path.isdir(mask_path_class):
            os.mkdir(mask_path_class)

    
        # before using the data, normalize them
        dataset_tmp = copy.deepcopy(dataset)
        dataset_tmp.raw = list(dataset_tmp.raw)
    
        # dataset normalization
        min_spectr = np.min([torch.min(torch.abs(torch.tensor(dataset_tmp[i][0]))) for i in range(len(dataset_tmp))])
        max_spectr = np.max([torch.max(torch.abs(torch.tensor(dataset_tmp[i][0]))) for i in range(len(dataset_tmp))])
    
        
        # select from dataset instances with label i
        dataset_tmp = dataset_tmp.select_class(i)
        print("Size of data set: ", len(dataset_tmp))

        for i, _ in enumerate(dataset_tmp):
            spectr = dataset_tmp.spectrograms[i]
            spectr = torch.tensor(spectr.real)
            spectr = torch.abs(spectr)
            spectr = (spectr - min_spectr) / (max_spectr - min_spectr)
            dataset_tmp.spectrograms[i] = spectr

        
   
        # for each element in the dataset, compute its mask
        # loop over each element in the dataset
        input = dataset_tmp[0][0]
        print("Shape of first element: ", dataset_tmp[0][0].shape, flush=True)

        image_size = tuple((input.shape[1], input.shape[2]))
        nchannels = input.shape[0]
        print("nchannels: ", nchannels, flush=True)

        # transform dataset to tensor
        spectrograms, raw_signals, labels, ids, channels = dataset_tmp[:len(dataset_tmp)]
        # size of input: (batch_size, nchannels, image_size[0], image_size[1])
        spectrograms = torch.stack(spectrograms).float()
        print("Shape of input: ", spectrograms.shape, flush=True)
        
        labels = torch.stack([torch.tensor(label) for label in labels]).long()
        ids = np.asarray(ids)
        channels = np.asarray(channels)

        single_mask_training(model, spectrograms, raw_signals, labels, ids, channels, lam, mask_path_class, image_size, nchannels, figures=CONFIG.save_figures)        

        del dataset_tmp

    
