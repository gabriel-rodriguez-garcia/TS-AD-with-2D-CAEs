import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import progressbar
import os


### Functions
##########################################

def import_residuals(folder_residuals):
    """Import residuals. Testing and Training residuals have to be in the same folder."""

    #Training Residuals
    errors_train_1 = np.load(os.path.join(folder_residuals,'train_errors_part 1.npy'))
    errors_train_2 = np.load(os.path.join(folder_residuals,'train_errors_part 2.npy'))
    errors_train = np.squeeze(np.concatenate((errors_train_1,errors_train_2),axis=0),axis=(1,2))

    #Testing Resiudals
    errors_test = np.squeeze(np.load(os.path.join(folder_residuals, 'test_errors.npy')), axis=(1, 2))

    return errors_train, errors_test


def plot_error_distribution(errors_train, errors_test):
    """Plot error distribution of encoded training and testing data"""

    n=np.shape(errors_train)[0]
    n_bins=int(np.sqrt(n))

    #Histogram training errors: Log Scale
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Training errors Log-scale', fontsize=25, color='black')
    hist_train,_,_=plt.hist(errors_train,bins=np.linspace(0,np.max(errors_train),n_bins),density=False, cumulative=False)
    plt.yscale('symlog')
    plt.show()


    #Histogram testing errors: Log Scale
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Testing errors Log-scale', fontsize=25, color='black')
    hist_test,_,_=plt.hist(errors_test,bins=np.linspace(0,np.max(errors_test),n_bins),density=False, cumulative=False)
    plt.yscale('symlog')
    plt.show()

    print('Max/Min error of training set is :', np.max(errors_train),np.min(errors_train))
    print('Max/Min error of testing set is :', np.max(errors_test),np.min(errors_test))

def compute_threshold(errors_train):
    """ Compute the decision threshold, which in this case is defined as the 99% quantile of the training distribution"""

    n = np.shape(errors_train)[0]
    n_bins = int(np.sqrt(n))

    # Threshold
    hist_errors_train, bins_errors_train, _ = plt.hist(errors_train,
                                                       bins=np.linspace(0, max(errors_train), n_bins * 5),
                                                       density=True, cumulative=True)
    plt.show()
    print('Max error of training set is :', np.max(errors_train))
    for i in range(10000):
        if hist_errors_train[i] > 0.99:
            print('Training 99% quantile:', bins_errors_train[i - 1])
            break

    thresh = np.round(bins_errors_train[i - 1])
    return thresh

def measure_performance(errors_test,thresh,encoding,path_ground_truth):
    """ Plot the ROC Curve to asssess the performance of the model """

    if encoding=='GS':
        entire_signal_errors_test = np.reshape(errors_test, (594, 108))
    else:
        entire_signal_errors_test = np.reshape(errors_test, (594, 120))

    entire_signal_errors_max_test = []
    anom_signal = []
    anom_signal_idx = []

    for i in range(0, 594):
        entire_signal_errors_max_test.append(np.max(entire_signal_errors_test[i, :]))

        if entire_signal_errors_max_test[i] > thresh:
            anom_signal.append(1)
            anom_signal_idx.append(i)
        else:
            anom_signal.append(0)

    print('number of anomalous signals in test set', np.sum(anom_signal), np.sum(anom_signal) / 594 * 100, '%')

    P = 297
    N = 297
    gt = np.array(pd.read_csv(path_ground_truth,usecols=['anomaly']))
    gt = np.reshape(gt, (1, 594))
    TPR = []  # TP/P = Correctly Classified as anomalous / Anomalous
    FPR = []  # FP/N = Falsly Classified as anomalous / Normal

    for threshold in progressbar.progressbar(range(0, int(np.max(np.concatenate((errors_train,errors_test),axis=0))))):
        anom_signal = []
        # Classify all time series
        for i in range(0, 594):
            if entire_signal_errors_max_test[i] > threshold:
                anom_signal.append(1)
            else:
                anom_signal.append(0)

        # Evaluation of True positive rate (TPR) and False positive rate (FPR)
        counts, bin_edges = np.histogram(np.array(anom_signal + gt)[0], bins=[-1, 0, 1, 2, 3])  # Count the 2's
        TP = counts[3]
        TPR.append(TP / P)

        counts, bin_edges = np.histogram(np.array(anom_signal - gt)[0], bins=[-1, 0, 1, 2, 3])  # Count the 1's
        FP = counts[2]
        FPR.append(FP / N)

    plt.plot(FPR, TPR, 'b', label=encoding)
    plt.plot(np.linspace(0, 1, 2), np.linspace(0, 1, 2), 'black', label='Rand. Numb. Gen.')
    plt.plot(FPR[int(thresh)], TPR[int(thresh)], 'ob', markersize=10)
    plt.suptitle('ROC Curve for 2D-CAE (Encoding: '+encoding+')', fontsize=15)
    plt.legend(loc='lower right', fontsize=15, title_fontsize=15, bbox_to_anchor=(1.97, 0.54), fancybox=True)
    plt.xlabel('FPR', fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    plt.show()


### Main
##########################################
if __name__== '__main__':


    errors_train,errors_test=import_residuals(folder_residuals='../Part3_Residuals')

    plot_error_distribution(errors_train=errors_train,errors_test=errors_test)

    thresh=compute_threshold(errors_train=errors_train)

    measure_performance(errors_test=errors_test,thresh=thresh,encoding='GS',path_ground_truth='/Users/gabrielrodriguez/Master 3/Semester Project/Classification/Ground Truth.csv')

