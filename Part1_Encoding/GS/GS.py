import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import progressbar
from scipy import signal
from skimage.measure import block_reduce
import os

# Functions
##############################################

def import_data(path_dataset):
    """Import data from hdf5 file"""

    X=np.array(pd.read_hdf(path_dataset))

    print('Data set shape:',np.shape(X))
    print('#####################################')
    
    return X

def encode_dataset(dataset,detail_factor):
    """ The encoding per se does not require any parameters to be tuned except for the detail_factor. 
        The value controlls how detailed the images are going to be. Increase the value for more detailed images. 
        (This is due to the rounding operator).
        The default value for the detail_factor is set to 255 as in the following paper: 
        "Online Fault Diagnosis Method Based on Transfer Convolutional Neural Networks" by G. Xu et al. 2019
        
        Please note that in the paper the mathematical definition is not correct. The rounding operator should be 
        applied after multiplying the detail_factor

        The slicing procedure can be modified if necessary.
        Again using the default parameters will lead to an output image size of 64x64.
        """
    # Extract the max and min of the dataset. This is required for the normalization.
    tot_max= np.max(dataset)
    tot_min= np.min(dataset)

    print('total max :',tot_max)
    print('total min :',tot_min)
    
    X_norm=dataset*0
    n=np.shape(dataset)[0]
    for i in range(n):
        signal=X[i,:]
        # Constant zero time series are not normalized
        if np.max(signal)==np.min(signal):
            X_norm[i,:]=np.zeros((1,l))
        else:
            X_norm[i,:]=(detail_factor*(signal-tot_min)/(tot_max-tot_min)).round(decimals=0)
        
    # Cut last 96 numbers in order to achieve an output shape of 64x64
    X_norm=np.delete(X_norm,range(61440-96,61440),axis=1)
    print(X_norm.shape)
    
    # Slice the data 
    slice_len=64
    X_gs=np.zeros((n*108,slice_len,slice_len))
    image=np.zeros((1,slice_len,slice_len))
    counter=0
    shift=0
    for p in range(n):
        shift=0
        for s in range(108):
            for i in range(slice_len):
                image[0,i,:]=X_norm[p,shift*8:shift*8+64]
                shift+=1

            X_gs[counter]=image
            counter+=1 
            
    print('Encoding successful!')
    print('#####################################')
    
    return X_gs

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
    
    X_gs=encode_dataset(dataset=X, detail_factor=255)
    X_test_gs=encode_dataset(dataset=X_test, detail_factor=255)
    
    save_data(directory='GS_Images',dataset_name='X_gs.npy',dataset=X_gs)
    save_data(directory='GS_Images',dataset_name='X_test_gs.npy',dataset=X_test_gs)

# Display Images
#############################################
X_gs=np.load('GS_Images/X_gs.npy')
sample_image=X_gs[200] 
plt.figure(figsize=(5, 5))
plt.imshow(sample_image,cmap='gray')
plt.colorbar(fraction=0.0457, pad=0.04)
# plt.clim(-1,1)
plt.show()

