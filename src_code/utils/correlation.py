"""
The functions provided in this file are used to compute the correlation between the essential frequencies represented by 
the masks for each channel. The correlation is computed using a novel Id based method.

The following code an adaption of code available @ https://github.com/lorenzobasile/ImplicitBiasAdversarial/tree/main
Which provides the implementation for the paper: Relating Implicit Bias and Adversarial Attacks
through Intrinsic Dimension, by L. Basile, N. Karantzas, A. D'Onofrio, L. Bortolussi, A. Rodriguez, F. Anselmi

"""

import numpy as np
import scipy
import scipy.stats
from tqdm import tqdm
from dadapy.data import Data
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from scipy.optimize import curve_fit

def cosine_sim(dataset1, dataset2):
    '''
    Function to compute cosine similarity between two data sets

    Param dataset1 (np.array): first data set to correlate
    Param dataset2 (np.array): second data set to correlate

    Return: mean and standard deviation of cosine similarities
    '''
    product=np.sum(dataset1*dataset2, axis=1)
    norm1=np.linalg.norm(dataset1, 2, axis=1)
    norm2=np.linalg.norm(dataset2, 2, axis=1)
    corrs=np.divide(np.divide(product, norm1), norm2)
    return np.mean(corrs), np.std(corrs)


""" def compute_id(dataset):
    '''
    Function to compute Intrinsic Dimension of a data set
    Input: data set (np.array)
    Return: intrinsic dimension of data set, estimated with TwoNN (Facco et al., 2017)
    '''
    # data object, container for all dadapy functions
    # maxk is the maximum number of neighbors to consider
    data = Data(dataset, maxk=2) 
    del dataset
    id=data.compute_id_2NN()[0]
    return id """

def compute_id(dataset, nrep=10):
    '''
    Function to compute Intrinsic Dimension of a data set
    Input: data set (np.array)
    Return: intrinsic dimension of data set, estimated with TwoNN (Facco et al., 2017)
    '''
    # data object, container for all dadapy functions
    # maxk is the maximum number of neighbors to consider
    data = Data(dataset, maxk=2) 
    del dataset
    
    return data.compute_id_2NN(decimation=0.9, n_iter=nrep)[:2]

""" def compute_id(data, algorithm='base', dists=None):
    nrep = 10
    id = np.zeros(nrep)
    nsamples =  int(data.shape[0]*0.8) # use part of the data
    n = data.shape[0]

    # compute pairwise distances
    if dists is None:
        dists = pairwise_distances(data, metric='euclidean')

    for i in range(nrep):
        # randomly sample nsamples points
        idx = np.random.choice(data.shape[0], nsamples, replace=False)
        
        twoNN = np.zeros((n, 2))

        for j in idx:
            dists_sorted = np.sort(dists[j,idx])
            twoNN[j,:] = dists_sorted[1:3] # discard the first element (distance 0), given by the point itself
        
        # for each point compute the ratio between r2 and r1
        # for sample data
        mu = np.zeros(nsamples)
        for k,j in enumerate(idx):
            mu[k] = twoNN[j,1]/twoNN[j,0]
        
        log_mu = np.log(np.sort(mu))

        if algorithm == "ml":
            id[i] = (n - 1) / np.sum(log_mu)
        else:
            y = -np.log(1 - np.arange(1, nsamples+1) / (nsamples+1))
            func = lambda x, m: m*x
            comp_id, _ = curve_fit(func, log_mu, y)
            id[i] = comp_id[0]

    error = np.std(id)
    return id, error """


def compute_correlation(dataset1, dataset2, file_path, raw=True):
    """
    Function to compute correlation between the essential frequencies represented by
    the masks for each channel or between raw signals themselves. 
    The correlation is computed using a novel Id based method.

    The ID for the dataset made of the concatenation of the two masks (or the two raw signals) is computed.
    Then, the ID is computed for datasets obtained by shuffling the order of first set of data.
    The Z-score and p-value are computed to assess the significance of the correlation.

    Input:
        dataset1 (MaskDataset): first dataset
        dataset2 (MaskDataset): second dataset
        file_path (str): path to the file where to store the results
        raw (bool): whether to use raw data or masks learned on spectrograms
    """


    # get indices of the masks of the two lobes
    _, ind1, ind2 = np.intersect1d(dataset1.id, dataset2.id, assume_unique=True, return_indices=True)
    
    if raw:
        dataset1.raw = np.array(dataset1.raw)
        dataset2.raw = np.array(dataset2.raw)

        data1 = dataset1.raw[np.array(ind1)]
        data2 = dataset2.raw[np.array(ind2)]       
        
        data1 = np.array(data1, dtype=np.float32)
        data2 = np.array(data2, dtype=np.float32)
        # remove dimensions 1
        if len(data1.shape) > 2:
            data1 = np.squeeze(data1, axis=1)
            data2 = np.squeeze(data2, axis=1)

    else:
        data1=np.array(dataset1.masks)[ind1.astype(int)]
        data2=np.array(dataset2.masks)[ind2.astype(int)]
        print(data1.shape, data2.shape)
        data1 = data1.reshape(-1, data1.shape[1]*data1.shape[2])
        data2 = data2.reshape(-1, data2.shape[1]*data2.shape[2])
        print(data1.shape, data2.shape)

    # if too few masks are provided, return 
    if len(data1) < 5:
        try:
            with open(file_path, 'a+') as f:
                f.write(f'Error: Too few matching data provided for correlation computation. Found only {len(data1)} in union\n')
            return 
        except:
            print(f'Error: Too few matching data provided for correlation computation. Found only {len(data1)} in union\n')
            return
        
    
    mu, sigma=cosine_sim(data1, data2)

    full_data=np.concatenate([data1, data2], axis=1)
    id, err = compute_id(full_data, nrep=5)
    noshuffle=id
    
    # compute Pearson coefficient
    pearson_coeff = np.corrcoef(data1.flatten(), data2.flatten())
    assert pearson_coeff.shape == (2, 2)
    pearson_coeff = pearson_coeff[0,1] 

    shuffle=[]
    zscore = 2
    nrepeat = 0
    while zscore > 1. and nrepeat < 300:
        for _ in tqdm(range(50)):
            permutation = np.random.permutation(data1)
            del full_data
            full_data=np.concatenate([permutation, data2], axis=1)
            id, _ = compute_id(full_data, nrep=1)
            shuffle.append(id)
            nrepeat += 1
        
        zscore = (noshuffle - np.mean(shuffle))/np.std(shuffle)
    
    shuffle=np.array(shuffle)
    pvalue=scipy.stats.norm.cdf(zscore)
    
    
    try:
        with open(file_path, 'a') as f:
            f.write(f'Number of data: {len(data1)}\n')
            f.write(f'Data shape: {data1.shape}\n')
            f.write(f'Pearson coefficient: {pearson_coeff}\n')

            f.write(f'Cosine similarity: mean {mu}, std {sigma}\n')
            f.write(f'No-shuffle Id: {noshuffle}\n')
            f.write(f'Shuffled Id mean: {shuffle.mean()}, std: {shuffle.std()}\n')
            f.write(f'Z-score: {zscore}\n')
            f.write(f'P-value: {pvalue}\n')
            f.write(f'---------------------------------------\n\n')
    except:
        print('Error: cannot write on file', flush=True)

    return zscore, pvalue, pearson_coeff




def read_correlation_table(path, results, raw, band_range, kind='zscore'):
    
    if kind == 'zscore':
        table = results['zscore']
        pvalues = results['pvalue']
        title = 'Zscores between raw signals' if raw else 'Zscores between masks'
        title += ' - ' + band_range 
        file_path = path[:-9]+'zscore.png'

    elif kind == 'pearson':
        table = results['pearson']
        # make table symmetric
        # table = table + table.T - np.diag(np.diag(table))
        title = 'Pearson Coefficient between raw signals' if raw else 'Pearson Coefficient between masks'
        title += ' - ' + band_range 
        file_path = path[:-9]+'pearson.png'
    
    else:
        print('Error: invalid kind of correlation', flush=True)
        return
    
    channels = results['channels']

    # plot correlation table for zscore and pvalue
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(table, cmap='Blues_r') if kind == 'zscore' else ax.imshow(table, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(channels)))
    ax.set_yticks(np.arange(len(channels)))
    ax.set_xticklabels(channels)
    ax.set_yticklabels(channels)
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Channel 1', fontsize=20)
    ax.set_ylabel('Channel 2', fontsize=20)
    # put zscore in each cell
    for i in range(len(channels)):
        for j in range(i, len(channels)):
            _ = ax.text(j, i, round(table[i, j], 2), ha="center", va="center", color="k", fontsize=60/len(channels))
            # and also pvalue in scientific notation
            if kind == 'zscore':
                _ = ax.text(j, i+0.3, '{:.1E}'.format(pvalues[i, j]), ha="center", va="center", color="k", fontsize=60/len(channels))

    plt.colorbar(im, ax=ax, shrink=0.8)
    # increase image resolution
    plt.savefig(file_path, dpi=300)
    plt.close()

    
