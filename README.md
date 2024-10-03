# Analysis of EEG during mental arithmetic tasks (EEG-MAT)


## Table of Contents

  - [Introduction](#introduction)
  - [Spatial correlation analysis](#spatial-correlation-analysis)
    - [Dataset](#dataset)
  - [Neural networks for mental workload classification](#neural-networks-for-mental-workload-classification)
    - [Feature extraction](#feature-extraction)
    - [Classification](#classification)
  - [Frequency masks](#frequency-masks)


## Introduction

This project aims to analyze EEG signals  of participants performing mental arithmetic tasks. The EEG signals were captured using a 21-channel EEG system. For each participant two recordings are available, one recorded while the participant was at rest and one recorded while the participant was performing the task. 

The first purpose of this project focused on the analysis of **spatial correlations** between the EEG signals.
An [Id-based method](https://www.nature.com/articles/s41598-017-11873-y) was used to calculate non-linear correlations between the EEG signals captured by different electrodes. 

A second part of this project focused on the training of a deep learning model to classify the EEG signals into different classes. 
EEG signals were transformed into images using the wavelet transform. 
Once one or more classifiers had been trained, the **frequency masks** representing the essential frequencies used by the model were obtained. The frequency masks are used to understand which frequency bands and which electrodes are most important for the classification of the EEG signals. This part of the project is based on the method proposed in the paper [Investigating Adversarial Vulnerability and Implicit Bias through Frequency Analysis](https://arxiv.org/pdf/2305.15203).

The entire code developed is available in the `src_code` folder and it is written in Python (3.9).
In the `notebooks` folder, some Jupyter notebooks are provided.
These notebooks can be used to understand how to work with EEG data (`read_eeg_data.ipynb`, `build_montage.ipynb`), how to visualize the results of correlation analysis on raw data (`correlation_coeff_raw_data.ipynb`) and how to plot the results obtained through frequency masks extraction (`class_masks_visualization.ipynb`).

In order to run the code, the libraries mentioned in the *requirements.txt* file must be installed. 

```bash
pip install -r requirements.txt
```


### Dataset

The dataset used in this project is the [EEG During Mental Arithmetic Tasks](https://physionet.org/content/eegmat/1.0.0/). The dataset contains EEG signals captured from 36 participants performing mental arithmetic tasks. The dataset contains two .edf files for each participant (e.g. Subject00_1.edf, Subject00_2.edf), one for the CQ task and one for the MS task. The dataset also contains a .csv file with the labels (Good or Bad counting quality) for each participant. 

The first record of the .edf file contains the EEG signals captured while the participant was at rest. The second record contains the EEG signals captured while the participant was performing the task, using a 21-channel EEG system.



## Spatial correlation analysis

Spatial correlation analysis can be performed on raw signals or on features extracted from the signals. In this project, spatial correlation analysis was performed on raw signals.

The Id-based correlation method used is applied to all the different frequency bands of the EEG signals. The correlation values are calculated for each pair of electrodes or on those required by the user. 
To run the code, use the following command:

```python
cd src_code
python compute_correlation.py -ns 36 -ct cq -ch FP1,PZ
```

The above command will calculate the correlation values for the CQ (counting quality) task for the FP1 and PZ electrodes, using the recordings of all the participants (36 subjects). The correlation values will be saved in the *results* folder specified. 
For each frequency band (delta, theta, alpha, beta, gamma, full_spectrum), two or three folders will be generated depending on the number of classes. Each folder will contain four files:
- *correlation.txt*: this file contains the correlation values for each pair of electrodes and the configuration parameters;
- *correlation_table.pkl*: a pickle file containing the tables for Pearson and Z-score correlation values;
- two *.png* files providing a visual representation of the correlation values.


The correlation can be computed between frequency masks as well.
If for each channel a frequency mask is available, the correlation between the frequency masks can be computed. The code can be run using the following command:

```python
python compute_correlation.py -ns 36 -ct cq -ch FP1,PZ --mask True 
```




## Neural networks for mental workload classification

### Feature extraction

For EEG signals classification, a deep learning model was used. The model has been trained on the features extracted from the EEG signals. Continuous wavelet transforms were applied to raw EEG signals to extract features and obtain two-dimensional images which were used to train the ResNet18 model. These inputs can be provided to the neural network as single data items, or they can be made by all the channels (electrodes) available. The second option provides the most reliable results as the channels are correlated between each other.

### Classification

The classification of the EEG signals was performed using a ResNet18 architecture. Classifications were performed for the CQ task, the MS task (2 classes respectively) and for both tasks combined (3 classes).

The code can be run using the following command:

```python
cd src_code
python get_classifier.py -ns 36 -ct cq
```

A list of channels to be used can be provided, by default they will be all the channels available.
Furthermore, they can be provided to the neural network as single data items, or they a single item can be made by all the channels provided. 
The results are saved in a predefined folder, containing the confusion matrix, the classification report and the best model obtained during the training. 

## Frequency masks extraction

Once a suitable classifier has been trained, the essential frequency used by the model can be extracted. The frequency masks are used to understand which frequency bands and which electrodes are the most important for the classification of the EEG signals.
The extraction of the frequency masks can be performed using the following command:

```python
cd src_code
python get_class_masks.py -ns 36 -ct cq
```

The results are saved in a predefined folder, containing the configuration file and a specific folder for each class. Each class folder contains the masks for each electrode (channel) saved in a .npy file, a .png file providing a visual representation of the mask is also saved.

A single mask for each class is trained, however it is also possible to train a single mask for each data item available. This second option requires a lot of computational power and time and it would probably necessitate much more data items to be trained on. The study of correlations between the masks of different electrodes trained as single data items did not provide any significant results. This second option can be run using `get_masks.py` script, but it is not recommended and it was left just for the sake of completeness.
