import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import progressbar
from scipy import signal
from skimage.measure import block_reduce
import os
from scipy.misc import imresize
from PIL import Image

def import_data(path_dataset):
    """Import data from hdf5 file"""

    X=np.array(pd.read_hdf(path_dataset))

    print('Data set shape:',np.shape(X))
    print('#####################################')
    
    return X

def slice_timeseries(n_slices,dataset):
    """ Cut each time series in the dataset into equally sized slices of size -> Length of time series/n_slices.
    The function returns a matrix with shape (nr. of slices, slice length)"""

    n,l=np.shape(dataset)

    X = np.reshape(dataset,(n*n_slices,l//n_slices))

    print('sliced data shape (nr. of slices, slice length):',np.shape(X))
    print('#####################################')
    
    return X

def encode_dataset(dataset,signal_type, pooling_function):
    """
        The function returns a 3D matrix.
        The new 3D matrix contains several 2D matrices, which correspond to the time series encodings.
        The order of the objects does not change, which means for example that the 23rd slice of the 
        input dataset corresponds to the 23rd matrix in the 3D Matrix.
        
        The images in this case are discrete scalograms. Output imgae shape is fixed at 64x64 """
    
    factor=8
    smoothness_factor=4
    widths = np.linspace(1,64,64)
    widths= 2**(widths/smoothness_factor)
    n=np.shape(dataset)[0]
    X_sc=np.zeros((n,64,64))
    
    for i in range(0,n):
        cwtmatr = signal.cwt(dataset[i,:], signal_type, widths)
        X_sc[i]=block_reduce(cwtmatr, block_size=(1, factor), func=pooling_function)
        
    print('Encoding successful!')
    print('#####################################')

    return X_sc

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
    
    X=slice_timeseries(n_slices=120,dataset=X)
    X_test=slice_timeseries(n_slices=120,dataset=X_test)
    
    X_sc=encode_dataset(dataset=X,signal_type=signal.ricker, pooling_function=np.mean)
    X_test_sc=encode_dataset(dataset=X_test,signal_type=signal.ricker, pooling_function=np.mean)

    save_data(directory='SC_Images',dataset_name='X_sc.npy',dataset=X_sc)
    save_data(directory='SC_Images',dataset_name='X_test_sc.npy',dataset=X_test_sc)


# Display Images
##############################################
X_sc=np.load('SC_Images/X_sc.npy')
sample_image=X_sc[0] 
plt.figure(figsize=(5, 5))
plt.imshow(sample_image,cmap='jet')
plt.colorbar(fraction=0.0457, pad=0.04)
# plt.clim(-1,3)
plt.show()

