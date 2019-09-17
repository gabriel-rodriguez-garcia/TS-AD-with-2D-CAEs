import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import time
import progressbar
from scipy import signal
from skimage.measure import block_reduce
import os
from scipy.misc import imresize
from PIL import Image

# Functions
##############################################

def import_data(path_dataset):
    """Import data from hdf5 file"""

    X=np.array(pd.read_hdf(path_dataset))

    print('Data set shape:',np.shape(X))
    print('#####################################')
    
    return X

def encode_dataset(dataset,sampling_freq_hz=1024,window='hanning',nperseg=126, nfft=None, noverlap=118, padded=False):
    """ The function returns a 3D matrix.
        The new 3D matrix contains several 2D matrices, which correspond to the time series encodings/images.
        The order of the objects does not change, which means for example that the 23rd slice of the 
        input dataset corresponds to the 23rd encoding in the 3D Matrix.
        
        The images in this case are spectrograms. Note for this encoding, slicing is done after encoding.
        Output Image shape is fixed at 64x64"""
    
    n,l=np.shape(dataset)
    print('Encoding started...')
    
    for p in range(0,n):
        
        if p==0:
            _,_,sample = signal.stft(dataset[p,:], fs=sampling_freq_hz,window=window,nperseg=nperseg, nfft=nfft, noverlap=noverlap, padded=padded)
            X_sp=np.zeros((n,64,np.shape(sample)[1]))
            
        _,_,spec = signal.stft(dataset[p,:], fs=sampling_freq_hz,window=window,nperseg=nperseg, nfft=nfft, noverlap=noverlap, padded=padded)
        spec = np.abs(spec)**2
        spec=np.where(spec==0.0, 0.0001, spec)
        spec=10*np.log10(spec) # dB Scale
        
        X_sp[p]=spec
        

    print('Encoding successful!')
    print('#####################################')

    return X_sp

def slice_images(n_slices,encoding_dataset):
    """ Cut each spectrogram into equally sized chunks of size 64x64
        Number of slices per encoding is equal to n_slices."""
    
    n=np.shape(encoding_dataset)[0]
    X_sp_red=np.zeros((n*n_slices,64,64))
    for p in range(0,n):
        j=0
        for i in range (p*n_slices,(p+1)*n_slices):
                X_sp_red[i] = encoding_dataset[p,:,j*64:(j+1)*64]
                j+=1

    print('output 3D Matrix shape:',np.shape(X_sp_red))
    print('#####################################')
    
    return X_sp_red

def save_data(directory,dataset_name,dataset):
    """ The datasets are simply saved as numpy matrices
        To import the numpy matrix again use np.load("<path_to_file>")"""
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    print('Saving data...')
    np.save(os.path.join(directory,dataset_name),dataset)
    print(dataset_name,' saved at: ',os.path.join(directory,dataset_name))


# Create encodings
##############################################

if __name__=='__main__':

    X = import_data(path_dataset='../Input_Data/dftrain.h5')
    X_test = import_data(path_dataset='../Input_Data/dfvalid.h5')
    
    X_sp=encode_dataset(dataset=X,sampling_freq_hz=1024,window='hanning',nperseg=126, nfft=None, noverlap=118, padded=False)
    X_test_sp=encode_dataset(dataset=X_test,sampling_freq_hz=1024,window='hanning',nperseg=126, nfft=None, noverlap=118, padded=False)
    
    X_sp=slice_images(n_slices=120,encoding_dataset=X_sp)
    X_test_sp=slice_images(n_slices=120,encoding_dataset=X_test_sp)
    
    save_data(directory='SP_Images',dataset_name='X_sp.npy',dataset=X_sp)
    save_data(directory='SP_Images',dataset_name='X_test_sp.npy',dataset=X_test_sp)


# Display Images
##############################################
X_sp=np.load('SP_Images/X_sp.npy')
sample_image=X_sp[0] 
plt.figure(figsize=(5, 5))
plt.imshow(sample_image,cmap='jet')
plt.colorbar(fraction=0.0457, pad=0.04)
# plt.clim(-1,1)
plt.show()

