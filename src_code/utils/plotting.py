import mne
from mne.channels import make_standard_montage, make_dig_montage
from scipy.signal import spectrogram
from utils.read_data import *
from utils.utils import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np




########################
# PLOTTING FUNCTIONS
########################



def plot_spectrogram(folder, dataset, subject=1, recording=4, channel='C1', nepoch=0, notitle=False, plot=True):
    """
    Function to show single channel spectrogram
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
    _,_, spectr = spectrogram(raw, fs=fs,  window='hann', nperseg=nperseg, noverlap=noverlap, scaling='spectrum', mode='magnitude', detrend=False)

    if plot:
        plt.imshow(spectr.squeeze(0), origin='lower', vmin=0)
        plt.colorbar(label='Amplitude(V)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        if not notitle:
            plt.title('Subject ' + str(subject) + ' recording ' + str(recording) + ' channel ' + str(channel) + ' epoch ' + str(nepoch))
        plt.show()

    return raw, spectr




def plot_raw(folder, dataset, subject=1, recording=4, channel=None, nepoch=None):
    """
    Plot raw data in dataset.
    Input:
        folder: folder containing the data
        dataset: an EEG custom dataset
        subject: subject ID (int)
        recording: recording ID (int)
        channel: channel name
            - if None, plot all channels
        epoch: epoch ID (int)
            - if None, plot first epoch
    """
    # get raw object
    raw, _ = get_raw(folder, dataset, subject, recording, nepoch)
    
    # select channel    
    channel = channel if channel is not None else 'all'
    if channel != 'all':
        raw = raw.pick([channel])

    
    # plot psd
    plt.figure(figsize=(20, 5))
    
    plt.plot(raw.times, raw.get_data().squeeze(0).T)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.title('Subject ' + str(subject) + ' recording ' + str(recording) + ' channel ' + str(channel) + ' epoch ' + str(nepoch), loc='right')
    plt.show()

    return raw



def plot_training_results(file_path):
    """
    Function to plot train and validation loss and accuracy
    for each epoch
    """

    text = open(file_path, 'r').read()

    # read results and plot them
    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for line in text.split('\n'):
        # skip non-numerical lines
        # if line starts with alphabet, skip
        if not line[0].isdigit() or len(line) == 0:
            continue
        # split line into 4 values
        line = line.split(',')
        # convert to float
        train_loss.append(float(line[1]))
        val_loss.append(float(line[2]))
        train_acc.append(float(line[3]))
        val_acc.append(float(line[4]))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss', color='orange', linewidth=2)
    plt.plot(val_loss, label='val loss', color='darkcyan', linewidth=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc', color='orange', linewidth=2)
    plt.plot(val_acc, label='val acc', color='darkcyan', linewidth=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.savefig(file_path[:-4] + '.png')
    plt.close()




def get_confusion_matrix(ytrue, ypred, path=None):
    """
    Function to plot confusion matrix
    Input:
        ytrue: true labels
        ypred: predicted labels
    """

    cm = confusion_matrix(ytrue, ypred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    # remove colorbar
    ax.collections[0].colorbar.remove()

    if path is not None:
        plt.savefig(path + '/confusion_matrix.png')
        plt.close()
    else:
        plt.show()

    return cm