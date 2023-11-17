"""
In this file, the code for training the masks which will provide the essential
frequencies for each image in the dataset is provided.
Only correctly classified images (spectrograms) will be used for training.

The following code an adaption of code available @ https://github.com/lorenzobasile/ImplicitBiasAdversarial/tree/main
Which provides the implementation for the paper: Relating Implicit Bias and Adversarial Attacks
through Intrinsic Dimension, by L. Basile, N. Karantzas, A. D'Onofrio, L. Bortolussi, A. Rodriguez, F. Anselmi

"""


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.masks import Mask, MaskedClf
from classifier.training import *
from neurodsp.timefrequency.wavelets import compute_wavelet_transform
from utils.utils import *
import time



def ess_train(base_model, spectrograms, raw_sigs, labels, ids, channels, lam, path, img_size, nchannels=20, figures=False):

    '''
    Function to train essential frequency masks

    Param base_model (torch.nn.Module): pre-trained classifier
    Param spectrograms (torch.tensor): tensor containing a batch of images representing spectrograms    
    Param raw_sigs (torch.tensor): tensor containing a batch of 1D eeg signals
    Param labels (torch.tensor): tensor containing a batch of image labels
    Param id (str): id of the current image to be processed
    Param channel (str): the channel from which the current image (spectrogram was obtained)
    Param lam (float): parameter governing l_1 regularization
    Param path (str): path to the folder used to store masks
    Param img_size (tuple): (height, width) of the images
    Param figures (bool): True if masks are to be stored after training, False otherwise
    Param raw_dict (dict): dictionary containing the raw eeg data, needed in order to retrieve spectrogram parameters
    '''

    # clean folder if it already exists
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    else:
        clean_folder(path)
        os.mkdir(path+"figures/")
    """

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda" if next(base_model.parameters()).is_cuda else "cpu")

    n = len(labels)
    labels = labels.to(device)
    spectrograms = spectrograms.to(device)
    if len(spectrograms.shape) > 4:
        spectrograms = spectrograms.squeeze(1)
    
    print("spectrograms shape: ", spectrograms.shape)

    raw_sigs = raw_sigs.to(device)
    base_out = base_model(spectrograms)

   
    losses=[[] for i in range(n)]

    # we consider only correctly classified images
    correct_cls=(np.where((torch.argmax(base_out, axis=1)==labels).cpu())[0])


    print("start training masks...", flush=True)    

    for i in range(n):
        folder = path+ str(labels[i].item()) 
        filename = str(ids[i]) + '_' + str(channels[i])
                    

        # if the file does not exist already and the image was correctly classified
        if i in correct_cls and not os.path.isfile(folder +"/"+ filename +".npy"):
            since = time.time()
            model = MaskedClf(Mask((nchannels, img_size[0], img_size[1])).to(device), base_model)
            for p in model.clf.parameters(): # freeze classifier parameters
                p.requires_grad=False

            model.mask.train()
            optimizer = torch.optim.Adam(model.mask.parameters(), lr=0.01)
            epoch = 0
            while True:
                out = model(spectrograms[i])
                l = loss(out, labels[i].reshape(1))
                penalty = model.mask.M.abs().sum()
                l += penalty*lam
                losses[i].append(l.item())
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                # mask entries are in [0,1]
                model.mask.M.data.clamp_(0.,1.)
                epoch += 1

                # train until convergence, for no less than 100 epochs and no more than 3000 epochs
                if (epoch>200 and abs(l.item()-np.mean(losses[i][-20:]))<1e-3) or epoch>3000:
                    correct=torch.argmax(out, axis=1) == labels[i]
                    #print('out: ', out, ' labels: ', labels[i], ' correct: ', correct, ' loss: ', l.item(), flush=True)
                    print(f'Training time for {i}th image: ', time.time()-since, ' epoch ', epoch, '- ', correct, flush=True)

                    if correct:
                        #  make directory to save results
                        figures_folder = path+"figures/"+str(labels[i].item())+"/"
                        folder = path+ str(labels[i].item()) 
                        if not os.path.isdir(folder):
                            os.mkdir(folder)
                        if not os.path.isdir(figures_folder):
                            os.mkdir(figures_folder)

                        # plot loss
                        if figures:
                            plt.figure(figsize=(30,20))
                            plt.plot(losses[i], linewidth=5)
                            plt.xlabel("Epoch", fontsize=20)
                            plt.ylabel("Loss", fontsize=20)
                            # make ticks larger
                            plt.xticks(fontsize=20)
                            plt.yticks(fontsize=20)
                            plt.savefig(figures_folder + str(ids[i][0]) +"_loss.png")
                            plt.close()

                        mask=model.mask.M.detach().cpu()
                        mask=mask.squeeze().numpy()
                        
                        input_channels = nchannels

                        if input_channels == 1:
                            filename = str(ids[i]) + '_' + str(channels[i])
                            save_figure(figures_folder, filename, spectrograms[i][0], raw_sigs[i][0], mask) if figures else None
                            # save mask
                            mask_ch = np.sum(mask[0], axis=1)
                            np.save(folder +"/"+ filename +".npy", mask_ch)
                        else:
                            for j in range(input_channels):
                                # sum mask across time dimension
                                mask_ch = np.sum(mask[j], axis=1)

                                # if mask is all zeros, skip
                                if np.sum(mask_ch) != 0:
                                    filename = str(ids[i][j]) + '_' + str(channels[i][j])
                                    save_figure(figures_folder, filename, spectrograms[i][j], raw_sigs[i][j], mask[j]) if figures else None
                                    # save mask
                                    np.save(folder +"/"+ filename +".npy", mask_ch)
                               

                        del mask
                    break



def save_figure(figures_folder, filename, spectrogram, raw_sig, mask):
    
    #img = spectrograms[i].squeeze().detach().cpu().numpy()
    spectrogram = spectrogram.detach().cpu().numpy()
    # get spectrogram params 
    #fs, nperseg, noverlap = set_spectrogram_params(raw_dict)
    
    # image reconstruction: spectr * mask 
    #_, _, spectrogram = scipy.signal.stft(raw_sigs[i][0], fs=fs, nperseg=nperseg, nfft = None, noverlap=noverlap, scaling='spectrum', boundary=None, padded=False)
    #spectrogram = compute_wavelet_transform(raw_sig, fs=500, n_cycles=5, freqs=freqs)
    img_recon = mask*spectrogram

    # plot orginal signal, corresponding spectrogram, learned mask
    # filtred spectrogram and reconstructed signal in a row
    _, axes = plt.subplots(1, 4, gridspec_kw={'width_ratios': [2, 2, 2, 2]}, figsize=(100, 20))

        
    # original signal
    axes[0].set_title('Original signal', fontsize=40, pad=30)
    axes[0].plot(raw_sig.detach().cpu().numpy(), color='steelblue', linewidth=5)
    axes[0].tick_params(labelsize=40)
    axes[0].yaxis.offsetText.set_fontsize(40)
    axes[0].set_xlabel('Time', fontsize=40)
    axes[0].set_ylabel('Amplitude', fontsize=40)

    # corresponding spectrogram
    axes[1].set_title('Original spectrogram', fontsize=40, pad=30)                
    ims = axes[1].imshow(np.abs(spectrogram), aspect='auto', origin='lower', extent=[0, 250, 0, 62])
    axes[1].tick_params(labelsize=40)
    axes[1].set_xlabel('Time', fontsize=40)
    axes[1].set_ylabel('Frequency', fontsize=40)
    cbar = plt.colorbar(ims, ax=axes[1])
    cbar.ax.tick_params(labelsize=40)
    cbar.ax.yaxis.offsetText.set_fontsize(30)

    # learned mask
    axes[2].set_title('Computed mask', fontsize=40, pad=30)
    ims = axes[2].imshow(mask,  aspect='auto', origin='lower', cmap='Blues', vmin=0, extent=[0, 250, 0, 62])
    axes[2].tick_params(labelsize=40)
    cbar = plt.colorbar(ims, ax=axes[2])
    cbar.ax.tick_params(labelsize=40)
    cbar.ax.yaxis.offsetText.set_fontsize(30)

    # filtered spectrogram
    axes[3].set_title('Filtered spectrogram', fontsize=40, pad = 30)
    ims = axes[3].imshow(np.abs(img_recon),  aspect='auto', origin='lower', extent=[0, 250, 0, 62])
    axes[3].tick_params(labelsize=40)
    axes[3].set_xlabel('Time', fontsize=40)
    axes[3].set_ylabel('Frequency', fontsize=40)
    cbar = plt.colorbar(ims, ax=axes[3])
    cbar.ax.tick_params(labelsize=40)
    cbar.ax.yaxis.offsetText.set_fontsize(30)


    
    """ # reconstructed signal
    axes[4].set_title('Signal reconstruction', fontsize=40, pad=30)
    # add dimension to the image
    img_recon = np.expand_dims(img_recon, axis=0)

    _, sig_rec = scipy.signal.istft(img_recon, fs, nperseg=nperseg, noverlap=noverlap, nfft = None, scaling='spectrum')

    axes[4].plot(sig_rec[0], color='steelblue')
    axes[4].tick_params(labelsize=40)
    axes[4].yaxis.offsetText.set_fontsize(40)
    axes[4].set_xlabel('Time', fontsize=40)
    axes[4].set_ylabel('Amplitude', fontsize=40)
    """

    plt.tight_layout()
    # set white space so that the same white space is present 
    # between all the subplots
    plt.subplots_adjust(wspace=0.5)

    plt.savefig(figures_folder + filename +".png")
    plt.close()

