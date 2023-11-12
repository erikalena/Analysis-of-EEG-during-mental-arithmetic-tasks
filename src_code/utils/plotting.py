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



def plot_spectrogram(raw_signal, channel, id):
    """
    Function to show single channel spectrogram
    Input:
        raw_signal: raw signal
        channel: channel name
        id: id of the recording
    """

    spectrogram = compute_wavelet_transform(raw_signal, fs=500, n_cycles=5, freqs=np.arange(10, 70, 2))

    plt.imshow(np.abs(spectrogram), origin='lower', vmin=0,  aspect='auto', extent=[0, 250, 10, 70])
    plt.colorbar(label='Amplitude(V)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    subject = int(id[7:9])
    epoch = id.split('_')[2]
    plt.title('Subject ' + str(subject) + ' recording - channel ' + str(channel) + ' epoch ' + str(epoch))
    plt.show()








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