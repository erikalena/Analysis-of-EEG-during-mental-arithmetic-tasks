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
    number_of_subjects: int = 36
    datset_size: int = 0
    batch_size: int = 32
    start_idx: int = 0
    end_idx: int = 0
    nclasses: int = 2
    classification: str = 'ms'
    model_path: str = './results_classifier/resnet18_20231116-093539/best_model_params.pt' #/resnet18_20231114-221314/best_model_params.pt'
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

    spectrograms, raw_signals, labels, ids, channels = next(iter(testloader))
    print("Shape of spectrograms: ", spectrograms.shape, flush=True)

    return model
    

if __name__ == "__main__":
    
    # load data from folder
    sample_data_folder = './eeg_data/'

    file_path = f'./saved_datasets/eeg_dataset_ns_{CONFIG.number_of_subjects}_ch_{CONFIG.input_channels}_nc_{CONFIG.nclasses}_{CONFIG.classification}_05sec.pkl'
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
    

    lam = 0.001
    mask_path = './results_masks/'

    print("Training masks...", flush=True)

    batch_size = CONFIG.batch_size
    start_idx = CONFIG.start_idx
    end_idx = batch_size + start_idx

    # before using the data, normalize them
    dataset_tmp = copy.deepcopy(dataset)
    dataset_tmp.raw = list(dataset_tmp.raw)

 
    for i, _ in enumerate(dataset_tmp):
        spectr = dataset_tmp.spectrograms[i]
        spectr = scipy.signal.resample(spectr, 30, axis=2)
        spectr = torch.tensor(spectr.real)
        dataset_tmp.spectrograms[i] = torch.abs(spectr)
        dataset_tmp.raw[i] = torch.tensor(dataset.get_raw(i))

    # for each element in the dataset, compute its mask
    # loop over each element in the dataset
    input = dataset_tmp[0][0]
    print("Shape of first element: ", dataset_tmp[0][0].shape, flush=True)

    image_size = tuple((input.shape[1], input.shape[2]))
    nchannels = input.shape[0]
    print("nchannels: ", nchannels, flush=True)

    while end_idx < len(dataset):
        spectrograms, raw_signals, labels, ids, channels = dataset_tmp[start_idx:end_idx]
        spectrograms = torch.stack(spectrograms).float()
        raw_signals = torch.stack(raw_signals)
        labels = torch.stack([torch.tensor(label) for label in labels]).long()
        ids = np.asarray(ids)
        channels = np.asarray(channels)

        print(start_idx, end_idx)
        ess_train(model, spectrograms, raw_signals, labels, ids, channels, lam, mask_path, image_size, nchannels, figures=CONFIG.save_figures)
        print("Last image processed: ", end_idx, flush=True)
        start_idx = end_idx
        end_idx = batch_size + start_idx

    del dataset_tmp

    
