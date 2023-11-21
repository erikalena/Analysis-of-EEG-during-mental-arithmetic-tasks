import numpy as np
import os
import mne
import pickle
import scipy
import torch
import time
from classifier.training import EarlyStopper, save_checkpoint
import classifier.models
import copy
from torch.utils.data import DataLoader
from torchvision import transforms
from neurodsp.plts import plot_time_series, plot_timefrequency
from neurodsp.timefrequency.wavelets import compute_wavelet_transform
from enum import Enum




def set_spectrogram_params(fs):
    
    # Parameters for spectrogram calculation,
    # these are set to be the same as in the pape

    desired_window_size = .5  # seconds #1
    nperseg = int(desired_window_size * fs)  # Convert seconds to points
    overlap_percentage = 0.95 #0.8
    noverlap = int(nperseg * overlap_percentage)

    return fs, nperseg, noverlap



class EEGDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for the EEG data
    Input:
        spectrograms: a list of spectrograms
        raw: a list of raw eeg data
        labels: a list of labels
        id: a list of subject ID and recording ID
        channel: a list of channel names
        info: a dictionary containing all the information about the dataset
    """

    def __init__(self, spectrograms, raw, labels, id, channel, info):
        self.spectrograms = spectrograms # spectrograms
        self.raw = np.array(raw) # raw eeg data
        self.labels = labels 
        self.id = id  # subject ID and recording ID
        self.channel = channel # channel name
        self.info = info
        

    def __len__(self):
        return len(self.spectrograms)
    
    def __getinfo__(self):
        # get info attached to eeg dataset, which are necessary 
        # to recover a bunch of information about the dataset
        return self.info
    
    def __getshape__(self):
        # get shape of the dataset
        return self.spectrograms[0].shape[2]

    def __getitem__(self, idx):
        
        return self.spectrograms[idx], self.raw[idx], self.labels[idx], self.id[idx], self.channel[idx]
    
    def get_spectrogram(self, idx):
        return self.spectrograms[idx]
    
    def get_raw(self, idx):
        return self.raw[idx]
    
    def get_label(self, idx):
        return self.labels[idx]
    
    def get_id(self, idx):
        return self.id[idx]
    
    def get_channel(self, idx):
        return self.channel[idx]
    
    
    def set_spectrogram(self, idx, spectrogram):
        self.spectrograms[idx] = spectrogram
    
    def set_raw(self, idx, raw):
        self.raw[idx] = raw

    def select_class(self, idx):
        """
        Select only the data of the specified class
        """
        # get indices of the channels to be selected
        indices = [i for i, label in enumerate(self.labels) if label == idx]
        # select only the desired channels
        spectrograms = [self.spectrograms[i] for i in indices] if len(self.spectrograms) > 0 else []
        raw = [self.raw[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        id = [self.id[i] for i in indices]
        channel = [self.channel[i] for i in indices]

        return EEGDataset(spectrograms, raw, labels, id, channel, self.info)
    

    def select_channels(self, ch):
        """
        A function which returns an EEGDataset object with only the selected channels

        """
        # get indices of the channels to be selected
        indices = [i for i, channel in enumerate(self.channel) if channel == ch]
        # select only the desired channels
        spectrograms = [self.spectrograms[i] for i in indices] if len(self.spectrograms) > 0 else []
        raw = [self.raw[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        id = [self.id[i] for i in indices]
        channel = [self.channel[i] for i in indices]

        return EEGDataset(spectrograms, raw, labels, id, channel, self.info)
    
    def filter_data(self, lowcut, highcut, order=2):
        """
        Filter the EEG data in a specific frequency range
        """
        raw = []
        # filter raw data
        for idx, _ in enumerate(self.raw):
            raw.append(filter_eeg_data(self.raw[idx], fs=500, lowcut=lowcut, highcut=highcut, order=order))

        return EEGDataset(self.spectrograms, raw, self.labels, self.id, self.channel, self.info)
        
        

    # remove an item from the dataset
    def remove_item(self, idx):
        del self.spectrograms[idx]
        del self.raw[idx]
        del self.labels[idx]
        del self.id[idx]
        del self.channel[idx]



def filter_eeg_data(data, fs, lowcut, highcut, order):
    """
    Filter the EEG data in a specific frequency range
    """
  
    # Filter data in a specific frequency range 
    # Calculate the Nyquist frequency
    nyquist = 0.5 * fs

    # Design the band-pass filter using the Butterworth filter design
    b, a = scipy.signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')

    # Apply the filter to the EEG data
    filtered_eeg_data = scipy.signal.lfilter(b, a, data)

    return filtered_eeg_data


def read_eeg_data(folder, data_path, input_channels, number_of_subjects = 10, type = 'ms', save_spec = True, channel_list = None):

    '''
    Create an EEGDataset object from the mnist dataset.
    Input:
        folder: path to the folder containing the data
        data_path: path to save the dataset
        input_channels: number of input channels
        number_of_subjects: number of subjects to be loaded 
        type: determine which data to be used depending on the task we want to classify
         - 'ms': use data to classify mental state (rest, count)
         - 'cq': use data to classify counting quality (good, bad)
         - 'both': use data to classify both mental state and counting quality (3 classes)
        save_spec: if True, save spectrograms, so that if they are not needed, the size of the dataset is reduced
        channel: if not None, select only the specified channel
    Output: 
        dataset: an EEGDataset object
    '''

    # if data_path exists, load dataset from there
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    
    # for each edf file in folder
    ids = []
    channels = []
    raw = []
    spectrograms = []
    labels = [] # 0 = rest, 1 = count_g, 2 = count_b
    
    subject_count_quality = [0,1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1]

    nchannels = 20
    channel_names = ['FP1','FP2', 'F3','F4','F7','F8','T3','T4','C3','C4','T5','T6','P3','P4','O1','O2','FZ','CZ','PZ','A2']

    
    for subj in range(number_of_subjects):
        if subj > 35:
            break
        filename1 =  'Subject' + "{0:0=2d}".format(subj) + '_1.edf'
        filename2 =  'Subject' + "{0:0=2d}".format(subj) + '_2.edf'

        files = [filename2] if type == 'cq' else [filename1, filename2]

        for file in files:
            
            print(file)

            id = file.split('.')[0]
            label = int(id.split('_')[1]) - 1
            subject = int(id[7:].split('_')[0])

            if type == 'cq' and label == 1 and subject_count_quality[subject] == 1:
                label = 0
            elif type == 'both' and label == 1 and subject_count_quality[subject] == 1:
                label = 2

            # read file
            data = mne.io.read_raw_edf(folder + file, preload=True, verbose=False)
            
            fs = data.info['sfreq']

            factor = 1
            segment_length = int(fs/factor) #int(fs*6)

            sample = data.get_data(0)[0]
            # pad sample so that it is divisible by 6
            #sample = np.append(sample, [sample[-1], sample[-1]])

            if (type == 'ms' or type == 'both') and label == 0:
                # keep the first minute of rest
                sample = sample[:60*int(fs)]

            n_segments = factor*60#int(len(sample)/segment_length)

            #print(segment_length, n_segments)

            for j in range(n_segments):
                
                img_eeg = []
                raw_eeg = []

                identifiers = []

                for i in range(len(data.ch_names)-1):
                    if channel_list is not None and channel_names[i] not in channel_list:
                        continue
                    sample = data.get_data(i)[0]
                    eeg_data = sample[j*segment_length:(j+1)*segment_length] 
                   
                    raw_eeg.append(eeg_data)

                    #_, _, spectr = scipy.signal.stft(eeg_data, fs=fs, nperseg=nperseg, nfft = None, noverlap=noverlap, scaling='spectrum', boundary=None, padded=False)
                    #spectr = np.abs(spectr.real)[:80,:]
                    #img_eeg.append(spectr)

                    if save_spec:

                        freqs = np.arange(1, 60, 2)
                        #sig = scipy.signal.resample(eeg_data, 500)
                        #mwt = compute_wavelet_transform(sig, fs=250, n_cycles=15, freqs=freqs)
                        ncycles = np.linspace(1,5,len(freqs))
                        mwt = compute_wavelet_transform(eeg_data, fs=500, n_cycles=ncycles, freqs=freqs)
                        mwt = scipy.signal.resample(mwt, 200, axis=1)
                        img_eeg.append(np.abs(mwt))
                       
                       
                    else:
                        img_eeg.append([])

                    ch = data.ch_names[i].split(' ')[1]
                    identifiers.append(f'{id}_{j}')

                    
                if input_channels == 1:
                    spectrograms.extend(np.array(img_eeg))
                    raw.extend(raw_eeg)

                    labels.extend([label]*nchannels) if channel_list is None else labels.extend([label]*len(channel_list))
                    ids.extend(identifiers)
                    channels.extend(channel_names) if channel_list is None else channels.extend(channel_list)

                else:
                    spectrograms.append(np.array(img_eeg))
                    raw.append(raw_eeg)
        
                    labels.append(label) # get label from event
                    ids.append(identifiers)
                    channels.append(channel_names) 

    dataset = EEGDataset(spectrograms, raw, labels, ids, channels, {})

    with open(data_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    return dataset
    



def build_dataloader(dataset, batch_size, train_rate=0.8, valid_rate=0.1, shuffle=True, resample=False, num_subjects = 10):
    """
    A function which provides all the dataloaders needed for training, validation and testing
    Input:
        dataset: a custom dataset
        batch_size: the batch size
        train_rate: the percentage of the dataset used for training
        valid_rate: the percentage of the dataset used for validation
        test_rate: the percentage of the dataset used for testing
        shuffle: whether to shuffle the dataset before splitting it
        resample: whether to resample the spectrograms to make them smaller
    """

    # build trainloader
    train_size = int(train_rate * len(dataset))
    valid_size = int(valid_rate * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    # before loading data into dataloader, normalize the data
    dataset_tmp = copy.deepcopy(dataset)

    if resample:
        # resample spectrograms
        for idx, _ in enumerate(dataset):
            # resample spectrogram
            spectrogram = dataset_tmp.get_spectrogram(idx)
            dataset_tmp.spectrograms[idx] = scipy.signal.resample(spectrogram, 100, axis=2)
            if idx == 0:
                print("Shape of spectrogram after resampling: ", dataset_tmp.spectrograms[idx].shape)

    # transform data to tensors if not already
    for idx in range(len(dataset_tmp.raw)):
        # transform to tensors
        dataset_tmp.spectrograms[idx] = torch.tensor(dataset_tmp.spectrograms[idx].real).float() if len(dataset_tmp.spectrograms) > 0  else dataset_tmp.spectrograms[idx]
        dataset_tmp.raw[idx] = torch.tensor(np.array(dataset_tmp.raw[idx])).float() 
        dataset_tmp.labels[idx] = torch.tensor(dataset_tmp.labels[idx]).long() 
        #self.channel[idx] = torch.tensor(self.channel[idx]).long()

    min_spectr = np.min([torch.min(torch.abs(dataset_tmp[i][0])) for i in range(len(dataset_tmp))])
    max_spectr = np.max([torch.max(torch.abs(dataset_tmp[i][0])) for i in range(len(dataset_tmp))])
    #print('max of spectrograms', max_spectr)

    # normalize spectrograms
    for idx, _ in enumerate(dataset):
        spectrogram = torch.abs(dataset_tmp.spectrograms[idx])
        dataset_tmp.spectrograms[idx] = (spectrogram- min_spectr) / (max_spectr - min_spectr)

    

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset_tmp, [train_size, valid_size, test_size])
    del dataset_tmp

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle) if valid_size > 0 else None
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)  if test_size > 0 else None

    return trainloader, validloader, testloader
