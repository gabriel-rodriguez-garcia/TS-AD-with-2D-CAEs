### TimeSeries-AnomalyDetection-with-2D-CAEs ###

# Code Structure

The code is divided into 4 parts:
Part 1: Create Encodings,
Part 2: Train the Network,
Part 3: Evaluate Residuals,
Part 4: Performance Assessement

The project is designed to be used in the following way:
In part 1 the user can choose one of the 6 available encodings and run the script to encode the training and testing data.
In part 2 the network gets trained using the encoded data, which has been created in Part 1.
In part 3 the residuals (=errors) of the encoding images from training and testing data are computed and saved.
In part 4 the errors are used to compute the decision threshold and a ROC Curve gets plotted.

# PART 1 - Create Encodings

For each of the following encodings an individual script is available to create them:
1. Gramian Angular Field (GAF)
2. Markov Transition Field (MTF)
3. Recurrence Plot (RP)
4. Spectrogram (SP)
5. Scalogram (SC)
6. Gray Scale Images (GS)

In general the default parameters in each script have been selected as to produce encodings of size 64x64, which is what the default network expects for training. Creating images of different size can for some encodings be non trivial (SP, SC and GS), therefore the user is expected to understand the code before modifying any parameters.

Creation times for each encoding based on default training and validation datasets:
Reference Machine: Mac Book Pro 2018, 16 GB RAM, 2.2 GHz Intel Core i7

1. GAF: Training set   -> 8 min ||
        Validation set -> 2.5 min
        
2. MTF: Training set   -> 12 min ||
        Validation set -> 3.5 min
        
3. RP:  Training set   -> 11 min ||
        Validation set -> 3 min
        
4. SP:  Training set   -> 1 min ||
        Validation set -> 0.3 min
        
5. SC:  Training set   -> 33 min ||
        Validation set -> 11 min
        
6. GS:  Training set   -> 0.3 min ||
        Validation set -> 0.1 min

# PART 2 - Train the Network

For basic modifications change the values of the argument parser variables. 

- Train a new model: If you want to train a completely new model then just make sure that mode='new_encoding' and run the main script. Also make sure to choose one of the available encodings. The encoded data has to be 

