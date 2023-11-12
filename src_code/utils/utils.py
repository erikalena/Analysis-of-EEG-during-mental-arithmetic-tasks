import torch
import mne
from mne.channels import make_standard_montage, make_dig_montage
from scipy.signal import spectrogram
from utils.read_data import EEGDataset
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import copy
import os
import pickle
import shutil
import scipy


##########################################
# utility functions to work with EEG data, 
# mainly based on MNE library
##########################################


def change_channel_names(raw):
    """
    make channel names compatible with those of other montages,
    and also more easy to deal with
    """
    # remove from raw channel names the final dots
    # and make all letters uppercase
    raw.rename_channels(lambda x: x.strip('.'))
    raw.rename_channels(lambda x: x.upper())

    return raw

def build_montage(raw):
    """
    Build a custom montage for the 64 channel 10-10 system
    """

    montage = make_standard_montage('biosemi64')

    # make a copy of montage positions
    montage_pos = copy.deepcopy(montage._get_ch_pos())
    # remove from montage_pos the channels that are not in raw
    del montage_pos['P9']
    del montage_pos['P10']
    # now add the channels which are missing from biosemi 64 channel system
    # which are T9 and T10
    # I have copied the positions from the 10-20 system
    montage_pos['T9'] = np.array([-0.0858941, -0.0158287, -0.048283 ])
    montage_pos['T10'] = np.array([ 0.0855599, -0.0163613, -0.048271 ])
    # make all keys in montage_pos uppercase
    montage_pos = {k.upper(): v for k, v in montage_pos.items()}
   
    raw = change_channel_names(raw)
   
    # Create a dictionary whose key are channel names and whose values are
    # the positions of the electrodes
    electrode_dicts = {ch_name: montage_pos[ch_name] for ch_name in raw.ch_names}

    # get fiducial points
    fid = montage.dig
    nasion = fid[1]['r']  # Nasion point
    lpa = fid[0]['r']  # Left point
    rpa = fid[2]['r']  # Right point

    custom_montage = make_dig_montage(nasion = nasion, lpa=lpa, rpa=rpa, ch_pos = electrode_dicts)

    return custom_montage
    


def get_events(raw):
    """ 
    Get events from raw data annotations
    """
    # define the event IDs explicitly based on your annotations
    event_id = {'T0': 0, 'T1': 1, 'T2': 2}
    events = []

    # loop through the annotations and manually create events
    for annot in raw.annotations:
        onset = int(annot['onset'] * raw.info['sfreq'])  # convert onset time to sample index
        events.append([onset, 0, event_id.get(annot['description'], 0)])  

    # convert the events list to a NumPy array
    events = sorted(events, key=lambda x: x[0])  # Sort events by onset time

    return events




def get_epochs(raw_dict):
    """
    Retrieve epochs from raw data and save them in a dictionary
    whose keys are made of a string containing the subject ID and
    the recording ID
    """

    reject_criteria = dict(
        eeg=150e-4, # unit: V
    )  
    event_id = {'T0': 0, 'T1': 1, 'T2': 2}

    # create a dictionary of epochs for each subject
    # and recording
    epochs_dict = {}

    for key, raw in raw_dict.items():
        events = get_events(raw)
        epochs = mne.Epochs(raw, events, event_id=event_id, preload=True, tmin=.1, tmax=4,
                            baseline=(None, .1), reject=reject_criteria, reject_by_annotation=True, verbose=False)
        epochs_dict[key] = epochs

    return epochs_dict




############################################
# utility functions to work with spectrograms
############################################

def set_spectrogram_params(raw_dict):
    # get channel names
    raw = list(raw_dict.values())[0]
    channel_names = raw.ch_names

    # get epochs
    epochs_dict = get_epochs(raw_dict)

    # set parameters for spectrogram calculation
    _, first_value = next(iter(epochs_dict.items()))
    time_vec = first_value.times   

    # Parameters for spectrogram calculation,
    # these are set to be the same as in the paper
    fs = 1.0 / (time_vec[1] - time_vec[0])  # Sampling frequency
    nsamples = len(time_vec)
    nperseg = int(nsamples/time_vec[-1])  # Window size for each segment, window size = 1s then 
    noverlap = nperseg*90/100  # in paper 90% overlap  

    desired_window_size = 1  # seconds
    nperseg = int(desired_window_size * fs)  # Convert seconds to points
    overlap_percentage = 0.9
    noverlap = int(nperseg * overlap_percentage)

    return fs, nperseg, noverlap


def get_raw(folder, dataset, subject, recording, nepoch):
    """
    Given dataset and desired subject, recording and epoch,
    return raw object and raw dictionary, which provides additional information
    for montage and channel names
    """

    raw_dict = read_data(folder, number_of_subjects=1, first_subj=subject)
    # put together in a string the subject and recording ID
    key = 'S' + str(f'{subject:03}') + 'R' + str(f'{recording:02}') + '_' + str(nepoch)
    
    # find in dataset all elements which have key has second element
    epoch = [dataset.get_raw(i) for i, _ in enumerate(dataset) if dataset.get_id(i) == key]

    # if epoch is empty, raise error
    if len(epoch) == 0:
        raise ValueError('Epoch not found')

    # make an array from the epoch
    epoch = np.array(epoch)
    epoch = epoch.squeeze(1) # 64 x 625
    epoch = mne.io.RawArray(epoch, info=dataset.__getinfo__())

    montage = build_montage(list(raw_dict.values())[0])
    epoch.set_montage(montage)

    return epoch, raw_dict


def get_stft(folder, dataset, subject=1, recording=4, channel='C1', nepoch=0, plot=True):
    """
    Function to show single channel short time fourier transform from raw data
    Input:
        folder: folder containing the data
        dataset: an EEG custom dataset
        subject: subject ID (int)
        recording: recording ID (int)
        channel: channel name
        epoch: epoch ID (int)
        plot: if True, plot the spectrogram
    """
    raw, raw_dict = get_raw(folder, dataset, subject, recording, nepoch)
    # select channel    
    raw = raw.pick([channel])
    raw = np.asarray(raw.get_data())
    fs, nperseg, noverlap = set_spectrogram_params(raw_dict)
    f, t, stft_eeg = scipy.signal.stft(raw, fs=fs, nperseg=nperseg, nfft = None, noverlap=noverlap, scaling='spectrum', boundary=None, padded=False)
    
    if plot:
        plt.imshow(np.abs(stft_eeg.squeeze(0)), vmin=0, origin='lower')
        plt.ylim([f[0], f[-1]])
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Subject ' + str(subject) + ' recording ' + str(recording) + ' channel ' + str(channel) + ' epoch ' + str(nepoch))

        cbar = plt.colorbar(label='Power Spectral Density')

        plt.show()

    return raw, stft_eeg, fs, nperseg, noverlap


def get_inv_stft(sftf_signal, fs, nperseg, noverlap, plot=True):

    _, inv_stft = scipy.signal.istft(sftf_signal, fs, nperseg=nperseg, noverlap=noverlap, nfft = None, scaling='spectrum', boundary=None)

    if plot:
        plt.figure()
        plt.plot(inv_stft[0])
        plt.plot(sftf_signal[0], color='red', alpha=0.8)
        plt.xlabel('Time [sec]')
        plt.ylabel('Signal')
        plt.show()
        plt.legend(['Original signal', 'Reconstructed signal'])

    return inv_stft



#################################################
# utility functions for managing datasets
#################################################


def get_dataset(first_subj, number_of_subjects, input_channels, lowcut, highcut, bandpass=True, raw=True, idx = 0):
    """
    Return the dataset of eeg data for the specified number of subjects starting from the first one.
    If bandpass is requested, the data is filtered in the specified band, otherwise the full spectrum is used.
    
    """

    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    # load data from folder
    sample_data_folder = '../eeg-motor-movementimagery/files/'

    # save results in a file
    dir_path = 'results_correlation/' 
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path += bands[idx] + '_band' if bandpass else 'full_spectrum'
    dir_path += '_masks/' if not raw else '/'

    # create directory if it does not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data_path = dir_path + 'MI_dataset_fs_' + str(first_subj) + '_ns_'+ str(number_of_subjects) +'_ch_' + str(input_channels)  + '.pkl'
    
    # if file saved_dataset.pkl exists, load it 
    if os.path.isfile(data_path):
        print(f'Loading dataset...', flush=True)
        dataset = load_dataset(data_path)
    else:
        print(f'Creating dataset...', flush=True)
        dataset = create_dataset(sample_data_folder, number_of_subjects, data_path=data_path, input_channels=input_channels, first_subj=first_subj, lowcut=lowcut, highcut=highcut)
        
    return dataset, dir_path


def filter_eeg_data(data, fs, lowcut, highcut, order):
    """
    Filter the EEG data in a specific frequency range
    """
    # Design a notch filter to remove the 60Hz component
    center_freq = 60.0/(fs/2)  # Center frequency of the notch filter
    Q = 30.0            # Quality factor
    b, a = scipy.signal.iirnotch(center_freq, Q)
    data = scipy.signal.filtfilt(b, a, data, axis=2)
    
    # Filter data in a specific frequency range 
    # Calculate the Nyquist frequency
    nyquist = 0.5 * fs

    # Design the band-pass filter using the Butterworth filter design
    b, a = scipy.signal.butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')

    # Apply the filter to the EEG data
    filtered_eeg_data = scipy.signal.lfilter(b, a, data)

    return filtered_eeg_data

def create_dataset(folder, nsubjects, data_path, input_channels = 1, first_subj=1, lowcut=0.5, highcut=40, order=2):

    """
    A function to create a dataset from the raw data
    Input:
        folder: folder containing the data
        nsubjects: number of subjects to be included in the dataset
        data_path: path to save the dataset
        input_channels: number of input channels
            - if 1, each channel will have one spectrogram which constitutes the input
            - if 64, the spectrograms will be batched together so that the input will have 64 channels
        first_subj: first subject to be included in the dataset (then proceed in ascending order)
        lowcut: low cut frequency for the band pass filter
        highcut: high cut frequency for the band pass filter
        order: order of the band pass filter
    """
    
    
    # check input channels is 1 or 64
    assert input_channels == 1 or input_channels == 64, "input_channels must be 1 or 64"
    
    # read data
    raw_dict = read_data(folder, nsubjects, first_subj)
    # get channel names
    raw = list(raw_dict.values())[0]
    channel_names = raw.ch_names

    # get epochs
    epochs_dict = get_epochs(raw_dict)

    # set parameters for spectrogram calculation
    _, first_value = next(iter(epochs_dict.items()))
    time_vec = first_value.times   

    # Parameters for spectrogram calculation,
    # these are set to be the same as in the paper
    fs = 1.0 / (time_vec[1] - time_vec[0])  # Sampling frequency
    """ nsamples = len(time_vec)
    nperseg = int(nsamples/time_vec[-1])  # Window size for each segment, window size = 1s then 
    noverlap = nperseg*90/100  # in paper 90% overlap   """

    desired_window_size = 1  # seconds
    nperseg = int(desired_window_size * fs)  # Convert seconds to points
    overlap_percentage = 0.9
    noverlap = int(nperseg * overlap_percentage)

    
    # create spectrograms
    img_eeg = []
    raw_eeg = []
    labels = []
    ids = []
    channels = []

    nchannels = len(channel_names)

    for key, epochs in epochs_dict.items(): # for each raw object
        for i in range(len(epochs)): # for each epoch 
            data = epochs[i].get_data()

            # remove noise and filter data
            data = filter_eeg_data(data, fs, lowcut, highcut, order)
            
            spectrograms = []
            
            for j in range(nchannels): # for each channel 
                #_,_, spectr = spectrogram(data[:, j, :], fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
                _, _, spectr = scipy.signal.stft(data[:, j, :], fs=fs, nperseg=nperseg, nfft = None, noverlap=noverlap, scaling='spectrum', boundary=None, padded=False) # function spectrogram substituted with stft
                spectrograms.append(spectr)

            if input_channels == 1:
                img_eeg.extend(np.array(spectrograms).squeeze(1))

                for j in range(nchannels): 
                    raw_eeg.append(data[:,j,:])

                labels.extend([epochs[i].events[:,-1].item()]*nchannels)
                ids.extend([str(key) + '_' + str(i)]*nchannels)
                channels.extend(channel_names)

            else:
                img_eeg.append(np.array(spectrograms).squeeze(1).transpose(1,2,0))
                raw_eeg.append(data.squeeze(0))
    
                labels.append(epochs[i].events[:,-1].item()) # get label from event
                ids.append(str(key) + '_' + str(i))
                channels.append(channel_names)


    # build dataset
    dataset = EEGDataset(img_eeg, raw_eeg, labels, ids, channels, raw.info)

    # if folder does not exist, create it
    dir_path = os.path.dirname(data_path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # save dataset
    with open(data_path, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset




def read_data(sample_data_folder, number_of_subjects, first_subj):
    """
    Read data from sample data folder, for each subject save 
    the recordings we are interested in in a dictionary
    """
    # create raw object for each subject and each run
    # save raw object in a dictionary
    raw_dict = {}

    # list of subjects that must be excluded
    subj_removed = [88, 89, 92, 100, 104]
    subj_selected = np.arange(first_subj, first_subj+number_of_subjects)
    # list of selected recordings
    selected = ['R04', 'R08', 'R12']
    line_frequency = 60.0  # remove power line noise

    for folder in os.listdir(sample_data_folder):
        if folder.startswith('S'):
            if int(folder[1:]) in subj_selected and int(folder[1:]) not in subj_removed:
                for file in os.listdir(sample_data_folder + folder):
                    if file.endswith('.edf') and file[-7:-4] in selected:
                        raw = mne.io.read_raw_edf(sample_data_folder + folder + '/' + file, preload=True, verbose=False)
                        #raw.resample(250., npad='auto')
                        # Apply the notch filter
                        #raw.notch_filter(freqs=line_frequency, picks='all')
                        raw_dict[file[:-4]] = raw 


                        
    # save montage for each raw object
    for key in raw_dict.keys():
        raw_dict[key].set_montage(build_montage(raw_dict[key]))

    return raw_dict



def load_dataset(data_path):
    """
    Load dataset from file
    """
    dataset = None
    if os.path.isfile(data_path):
        with open(data_path, 'rb') as f:
            print(data_path, flush=True)
            dataset = pickle.load(f)
        return dataset
    else:
        print("Error: file not found", flush=True)
        exit(0)



##################################################
# Utility functions for training and saving models
##################################################


def save_dataloaders(dataloaders, file_path):
    """
    Save dataloaders to file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(dataloaders, f)

def load_dataloaders(file_path):
    """
    Load dataloaders from file
    """
    with open(file_path, 'rb') as f:
        dataloaders = pickle.load(f)
    return dataloaders


##################################################
# other utility functions, probably to be removed
##################################################


def clean_folder(path):

    for filename in os.listdir(path):
        # iterate over all files in the subdirectory
        file_path = os.path.join(path, filename)
        try:
            # delete the file
            shutil.rmtree(file_path)
        except OSError as e:
            print("Error: %s : %s" % (file_path, e.strerror))


def create_lobe_dictionary(ch_names):
    """
    Function that creates a dictionary with the 5 lobes as 
    keys and the channels belonging to each lobe as values

    """

    dict_ch = {'frontal': [], 'central': [], 'parietal': [], 'temporal': [], 'occipital': []}

    for ch in ch_names:
        # if ch name init with A or contains only letter F
        if ch[0] == 'A' or ch[0] == 'F':
            if ch[1] == 'C':
                dict_ch['central'].append(ch)
            elif ch[1] == 'T':
                continue
            else:
                dict_ch['frontal'].append(ch)
        elif ch[0] == 'C':
            dict_ch['central'].append(ch)
        elif ch[0] == 'P':
            if ch[1] == 'O':
                dict_ch['occipital'].append(ch)
            else:
                dict_ch['parietal'].append(ch)
        elif ch[0] == 'T':
            dict_ch['temporal'].append(ch)
        elif ch[0] == 'O':
            dict_ch['occipital'].append(ch)
        else:
            continue

    return dict_ch


