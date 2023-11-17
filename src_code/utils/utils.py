import torch
import mne
from mne.channels import make_standard_montage, make_dig_montage
from scipy.signal import spectrogram
from utils.read_data import EEGDataset
from utils.plotting import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import copy
import os
import pickle
import shutil
import scipy
from mne.viz.topomap import _get_pos_outlines


##########################################
# utility functions to work with EEG data, 
# mainly based on MNE library
##########################################

def get_montage(raw):
    # montage available in mne
    montage = mne.channels.make_standard_montage('standard_1020')

    # make a copy of montage positions
    montage_pos = copy.deepcopy(montage._get_ch_pos())

    montage_pos = {k.upper(): v for k, v in montage_pos.items()}
    electrode_dicts = {ch_name: montage_pos[ch_name] for ch_name in raw.info['ch_names']}

    # get fiducial points
    fid = montage.dig
    nasion = fid[1]['r']  # Nasion point
    lpa = fid[0]['r']  # Left point
    rpa = fid[2]['r']  # Right point

    custom_montage = mne.channels.make_dig_montage(nasion = nasion, lpa=lpa, rpa=rpa, ch_pos = electrode_dicts)

    return custom_montage

    
def get_topographic_map(filename, size):
    """
    filename: str
        filename of the edf file
    size: int
        number of data points in the time interval to plot
    """
    # read file from folder
    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)

    # standardize channel names
    raw.rename_channels(lambda x: x.upper()[4:])

    # remove ECG channel
    raw.drop_channels(['ECG'])
    # and name A2-A1 as A2
    raw.rename_channels({'A2-A1': 'A2'})

    # set montage
    custom_montage = get_montage(raw)
    raw.set_montage(custom_montage)
    
    # get 2d positions of electrodes
    pos_2d = _get_pos_outlines(raw.info, picks=None, sphere=None)[0]
    positions = np.zeros((len(raw.ch_names), 2))
    for ch in raw.ch_names:

        idx = raw.ch_names.index(ch)
        positions[raw.ch_names.index(ch), :] = [pos_2d[idx][0], pos_2d[idx][1]-0.02]

    # extract data to plot
    data = np.zeros((len(raw.ch_names), size))

    
    for i, ch in enumerate(raw.ch_names):
        data[i, :] = raw[ch][0][0][:size]


    plot_topographic_map_time(data, positions, size)

    # plot topographic map gamma band
    psd_gamma, freqs = mne.time_frequency.psd_array_welch(raw.get_data(), sfreq=500, fmin=35, fmax=60, n_fft=500, n_overlap=250)
    psd_alpha, freqs = mne.time_frequency.psd_array_welch(raw.get_data(), sfreq=500, fmin=0.5, fmax=4, n_fft=500, n_overlap=250)

    # get one average psd for each channel
    psd_gamma = np.mean(psd_gamma, axis=1)
    psd_alpha = np.mean(psd_alpha, axis=1)
    #ratio = np.log2(psd_alpha/psd_gamma)
    plot_topographic_map_freq(psd_gamma, positions)

    return raw




############################################
# utility functions to work with spectrograms
############################################



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


