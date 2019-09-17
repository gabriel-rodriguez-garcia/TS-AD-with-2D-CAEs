### TS-AD-with-2D-CAEs ###

# Code structure

The code is divided into 4 parts:
part 1: Create Encodings and save them in a directory
part 2: Train the network
part 3: Evaluate the errors
part 4: Measure the performance

# PART 1

For each of the following encodings an individual script is available to create them:
1. Gramian Angular Field (GAF)
2. Markov Transition Field (MTF)
3. Recurrence Plot (RP)
4. Spectrogram (SP)
5. Scalogram (SC)
6. Gray Scale Images (GS)

In general the default parameters in each script have been selected as to produce encodings of size 64x64, which is what the default network expects for training. Creating images of different size can in some cases be non trivial, therefore the user is expected to understand the code before modifying any parameters.

Creation times for each Encoding based on default training and validation datasets:
Reference Machine: Mac Book Pro 2018, 16 GB RAM, 2.2 GHz Intel Core i7

1. GAF: Training set   -> 8 min
        Validation set -> 2.5 min
        
2. MTF: Training set   -> 12 min
        Validation set -> 3.5 min
        
3. RP:  Training set   -> 11 min
        Validation set -> 3 min
        
4. SP:  Training set   -> 
        Validation set -> 
        
5. SC:  Training set   -> 
        Validation set -> 
        
6. GS:  Training set   ->
        Validation set -> 



