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

- Train a new model: If you want to train a completely new model then just make sure that mode='new_encoding' and run the main script. Also make sure to choose one of the available encodings. Here is an example:

argp = parser.parse_args( <br/>
    ['--path_data','../Part1_Encoding', <br/>
     '--mode','new_training', <br/>
     '--dataset','training', <br/>
     '--cycles','500000', <br/>
     '--conv_kernel_size_1','4', <br/>
     '--conv_stride_1','2', <br/>
     '--pool_kernel_size','2', <br/>
     '--pool_stride','2', <br/>
     '--nr_channels_1','32',  <br/>
     '--bottleneck_size','160', <br/>
     '--batch_size','100', <br/>
     '--batch_size_testing','50', <br/>
     '--performance_eval_steps','10', <br/>
     '--checkpoint_save_steps','10000', <br/>
     '--encoding','GAF'])
     
  In this case a new model is trained on GAF data. The model will perform 50'000 gradient descent steps using 100 encoding images in each iteration. The loss gets computed after every 10 iterations and a checkpoint of the graph is saved after every 10'000 iterations. To inspect the loss or the graph use tensorboard by running the following command in terminal: <br/>
 tensorboard --logdir = < path to tensorboard summary >
        
 - Continue to train a model: If you want to continue to train a model whose checkpoint has been saved before, then just change mode='continue_training'. Note that all other parameters have to be equal to the ones saved in the checkpoint.
 
 <br/>
 
- Test a model: If you want to visually inspect if the reconstructions are good enough then just change mode='testing'. Also choose to either inspect training or testing images. The batch_size_testing value determines how many images will be displayed. Here is an example:

argp = parser.parse_args( <br/>
    ['--path_data','../Part1_Encoding', <br/>
     '--mode','testing', <br/>
     '--dataset','training', <br/>
     '--cycles','500000', <br/>
     '--conv_kernel_size_1','4', <br/>
     '--conv_stride_1','2', <br/>
     '--pool_kernel_size','2', <br/>
     '--pool_stride','2', <br/>
     '--nr_channels_1','32',  <br/>
     '--bottleneck_size','160', <br/>
     '--batch_size','100', <br/>
     '--batch_size_testing','50', <br/>
     '--performance_eval_steps','10', <br/>
     '--checkpoint_save_steps','10000', <br/>
     '--encoding','GAF'])
     
In this case 50 random images from the training data will be reconstructed and plotted.

# Part 3 - Compute the Residuals

The Residuals.py script is similar to to the main.py script of Part 2. Again for basic modifications only the argument parser variables have to be adjusted. The goal here is to compute the residuals of the training and testing reconstructions. Both are used in Part 4 to measure the model performance.

- Compute training residuals:
In order to compute all the residuals stemming from the training data, the script has to be run twice. In the first run setting the argument parser variable part='part 1' and in the second run setting part='part 2' respectively. This was required due to limited RAM of the machine running the script.
For buth runs set the variable dataset='training'.

Note the graph parameters should not change from the ones saved in the checkpoint in Part 2.

- Compute testing residuals:
In order to compute the residuals stemming from the testing data, the script can be run once by setting dataset='testing'.

# Part 4 -  Assess Model Performance

In order to assess the model performance, make sure to have the ground truth saved in a specific folder and specify the directory when calling the function:<br/>
measure_performance( ... ,path_ground_truth= < path to ground truth >.csv)

    
        
       


