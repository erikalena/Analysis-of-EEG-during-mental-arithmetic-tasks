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




def plot_topographic_map_freq(data, positions):

    fig = plt.figure(figsize=(20,10))
    fig, ax1= plt.subplots(ncols=1)
    im, _ = mne.viz.plot_topomap(data, positions, ch_type='eeg', axes=ax1,  cmap="viridis", size=5, show=False, names=raw.ch_names);
    ax_x_start = 0.95
    ax_x_width = 0.03
    ax_y_start = 0.0
    ax_y_height = 0.9
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title('dB',fontsize=15) # title on top of colorbar
    plt.show()
    
def plot_topographic_map_time(data, positions, size):

    names = None #raw.ch_names
    # plot topographic map
    fig = plt.figure(figsize=(50,10))
    fig,(ax1,ax2, ax3, ax4) = plt.subplots(ncols=4)
    idx1, idx2, idx3, idx4 = size//4, size//2, 3*size//4, size-1
    im,cm   = mne.viz.plot_topomap(data[:,idx1], positions, ch_type='eeg', cmap='jet', axes=ax1, show=False, names=names)
    ax1.set_title(f'{2*idx1} ms')
    im,cm   = mne.viz.plot_topomap(data[:,idx2], positions, ch_type='eeg', cmap='jet', axes=ax2, show=False, names=names)   
    ax2.set_title(f'{2*idx2} ms')
    im,cm   = mne.viz.plot_topomap(data[:,idx3], positions, ch_type='eeg', cmap='jet', axes=ax3, show=False, names=names)
    ax3.set_title(f'{2*idx3} ms')
    im,cm   = mne.viz.plot_topomap(data[:,idx4], positions, ch_type='eeg', cmap='jet', axes=ax4, show=False, names=names)   
    ax4.set_title(f'{2*idx4} ms')
    # manually fiddle the position of colorbar
    ax_x_start = 0.95
    ax_x_width = 0.02
    ax_y_start = 0.3
    ax_y_height = 0.4
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title('uV',fontsize=15) # title on top of colorbar
    plt.show()