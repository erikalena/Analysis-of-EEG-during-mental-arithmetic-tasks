import os
import sys
import datetime
import numpy as np
import pickle
import torch
from dataclasses import dataclass, field
from utils.correlation import read_correlation_table, compute_correlation
from utils.masks import MaskDataset
from eeg_mat.read_data import read_eeg_data

@dataclass
class Config:
    """
    A class to store all the configuration parameters
    """
    curr_time: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    number_of_subjects: int = 36
    nclasses: int = 2
    classification: str = 'ms'
    nelectrodes: int = 20
    input_channels: int = 1
    bandpass: bool = False
    raw: bool = True
    channels: list = field(default_factory=lambda: ['FP1','FP2', 'F3','F4','F7','F8','T3','T4','C3','C4','T5','T6','P3','P4','O1','O2','FZ','CZ','PZ','A2'])
    

    def save_config(self, file_path):
        """
        Write all the configuration parameters to file
        """
        with open(file_path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f'{key}: {value}\n')


CONFIG = Config()


def print_usage():
    print("Usage: python get_correlation.py <number_of_subjects> <channels>", flush=True)
    print("number_of_subjects: number of subjects to consider (default: 5)", flush=True)
    print("""channels: channels for which to compute the masks (e.g.: 'C3', 'C4')
            - if no channel is specified, correlation is computed for a list of predefined pairs of channels
            - no more than two channels must be entered by command line""", flush=True)
    sys.exit(0)


def save_correlation(dataset, channels, nclasses, raw, dir_path):
    """
    Save correlation results for the specified channels.
    Input:
        dataset: the dataset to use
        channels: the channels for which to compute the correlation
        nclasses: the number of classes
        raw: if True, use raw data, otherwise use masks
        dir_path: the directory where to save the results
    """
    results_path = dir_path + 'correlation.txt'

    # extract the band we are working on from the directory path
    band = dir_path.split('/')[3]

    CONFIG.save_config(results_path)

    # for each channel pair, compute the correlation
    table_zscore = np.zeros((len(channels), len(channels)))
    table_pvalue = np.zeros((len(channels), len(channels)))
    table_pearson = np.zeros((len(channels), len(channels)))

    for i in range(len(channels)):
        for j in range(i+1, len(channels)):
            print(f'Computing correlation between {channels[i]} and {channels[j]}')

            # if not computed already, compute correlation
            if table_zscore[i, j] != 0:
                continue

            with open(results_path, 'a') as f:
                f.write(f'\n\nComputing correlation for {channels[i]} and {channels[j]}\n\n')

            ch1, ch2 = channels[i], channels[j]

            # if raw is true, use raw data, otherwise use masks
            if raw:
                dataset1 = dataset.select_channels(ch1)
                dataset2 = dataset.select_channels(ch2)
            else:
                mask_path = './masks_v2/'
                # get the masks for the two channels
                dataset1 = MaskDataset(ch=ch1, path=mask_path, nclasses=nclasses)
                dataset2 = MaskDataset(ch=ch2, path=mask_path, nclasses=nclasses)

            with open(results_path, 'a') as f:
                f.write(f'Dataset1 size: {len(dataset1)}\n')
                f.write(f'Dataset2 size: {len(dataset2)}\n\n')
            
            zscore, pvalue = None, None
            if len(dataset1) > 1 and len(dataset2) > 1:
                zscore, pvalue, pearson = compute_correlation(dataset1, dataset2, file_path = results_path, raw=raw)
            
                table_zscore[i, j] = zscore.item()
                table_pvalue[i, j] = pvalue.item()
                table_pearson[i, j] = pearson


    # save tables as pkl in a file along with the channels
    table_path  = dir_path + 'correlation_table.pkl'
    # create an object with tables and channels
    info = {'zscore': table_zscore, 'pvalue': table_pvalue, 'pearson': table_pearson, 'channels': channels}

    with open(table_path, 'wb') as f:
        pickle.dump(info, f)

    # generate plot
    read_correlation_table(table_path, info, raw, band_range=band.replace('_', ' '), kind='zscore')
    read_correlation_table(table_path, info, raw, band_range=band.replace('_', ' '), kind='pearson')





def selected_channels(dataset, args):

    channels = []
    # get channel names
    ch_names = dataset.info.ch_names
    selected_channels = args[1:]

    # check if the channels are valid
    for ch in selected_channels:
        if ch not in ch_names:
            print(f"Error: channel {ch} is not valid.", flush=True)
            print_usage()
        else:
            print(ch)
            channels.append(ch) # substitute default list with the ones provided by user

    print('Selected channels: ', channels, flush=True)

    return channels


def get_dataset(input_channels, dir_path):
    """
    Return the dataset of eeg data for the specified number of subjects starting from the first one.
    """
    # load data from folder
    sample_data_folder = './eeg_mat/eeg_data/'

    print(f"Saving results in {dir_path}", flush=True)

    data_path = f'{dir_path}eeg_mat_ch_{str(input_channels)}_ns_{str(CONFIG.number_of_subjects)}_{str(CONFIG.classification)}_05sec.pkl'
    
    dataset = read_eeg_data(sample_data_folder, data_path, input_channels=input_channels, number_of_subjects=CONFIG.number_of_subjects, type = CONFIG.classification, save_spec=False)

    return dataset

def correlation_per_class(dataset, dir_path):
    """
    Compute correlation separately for each class of the dataset.
    """
    for i in range(CONFIG.nclasses):
        print(f"Computing correlation for class {i}", flush=True)
        print(len(dataset), len(dataset.raw), flush=True)
        dataset.raw = np.array(dataset.raw)
        print(dataset.raw[0].shape, flush=True)
        dataset_i = dataset.select_class(i)
        dir_path_i = dir_path + f'class_{i}/'

        print(dataset_i.raw[0].shape, flush=True)
        if not os.path.exists(dir_path_i):
            os.makedirs(dir_path_i)
        save_correlation(dataset_i, CONFIG.channels, CONFIG.nclasses, raw=CONFIG.raw, dir_path=dir_path_i)

    

if __name__ == "__main__":
    
    band_ranges = [0.5, 4, 8, 12, 30, 60]
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    # check CONFIG to be coherent, if raw is false, compared between masks 
    # which are learned on spectrograms (the full spectrum) so bandpass must be false
    if not CONFIG.raw:
        CONFIG.bandpass = False

    assert CONFIG.input_channels == 1, "Error: input channels must be 1 because correlation is checked between single channels"
    
    if len(sys.argv) > 4:
        print("Error: too many arguments provided.\n\n", flush=True)
        print_usage()
    
    sample_data_folder = './eeg_mat/eeg_data/'

    dir_path = './eeg_mat/results_correlation/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # get dataset
    dataset = get_dataset(CONFIG.input_channels, dir_path)
    CONFIG.datset_size = len(dataset)
    
    # compute correlation for each band
    
    """ if CONFIG.raw:
        for idx in range(len(band_ranges)-1):
            print(f"Computing correlation for band {bands[idx]}", flush=True)
            lowcut = band_ranges[idx]
            highcut = band_ranges[idx+1]
            
            # get dataset
            #dataset = get_dataset(CONFIG.input_channels, lowcut, highcut, bandpass=CONFIG.bandpass, raw=CONFIG.raw, idx=idx)
            
            # filter dataset in the specified band
            filtered_dataset = dataset.filter_data(lowcut, highcut)
            CONFIG.datset_size = len(filtered_dataset)

            # change channels if arguments are provided
            if len(sys.argv) > 2:
                CONFIG.channels = selected_channels(filtered_dataset, sys.argv) 

            band_dir_path = dir_path + bands[idx] + '_band' 
            band_dir_path += '_masks/' if not CONFIG.raw else '/'
            

            # split dataset in more datasets one for each class 
            correlation_per_class(filtered_dataset, band_dir_path)  """
   
    # compute correlation for the full spectrum
    band_dir_path = dir_path +  '/full_spectrum/'
    if not os.path.exists(dir_path):
        os.makedirs(band_dir_path) 
    
    #correlation_per_class(dataset, band_dir_path)

    print(f"Computing correlation for the whole dataset", flush=True)
    print(len(dataset), len(dataset.raw), flush=True)
    dataset.raw = np.array(dataset.raw)

    dir_path = dir_path + f'class_both/'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    save_correlation(dataset, CONFIG.channels, CONFIG.nclasses, raw=CONFIG.raw, dir_path=dir_path)

    
    


            
