import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os
import sys

from keras.models import Model
from keras.layers import Dense,Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, UpSampling2D

from Part2_Training.Prepare_Batch import Batch
from Part2_Training.Special_Layers import Fold, Unfold
from Part2_Training.Training_Schedule import Model

### Global Parameters
########################################################

# Change Parameters here and use them everywhere in the code using: argp.<variable_name>

parser = argparse.ArgumentParser()

# Paths/Directories
parser.add_argument("--path_data", type=str, help="Path to folder of input data")

# Work Schedule

parser.add_argument("--mode", choices=["new_training","continue_training","testing"])
parser.add_argument("--dataset", choices=["training","testing"])
parser.add_argument("--cycles",                type=int, help="Number of gradient descent steps")
parser.add_argument("--performance_eval_steps",type=int, help="Interval: number of steps to compute the loss")
parser.add_argument("--checkpoint_save_steps", type=int, help="Interval: number of steps to create a checkpoint of the model's state")
parser.add_argument("--batch_size_testing",    type=int, help="Number of images to show when testing")
parser.add_argument("--batch_size",            type=int, help="Number of images to use for computing loss while training")

# Architecture parameters

parser.add_argument("--conv_kernel_size_1",    type=int)
parser.add_argument("--conv_stride_1",         type=int)
parser.add_argument("--pool_kernel_size",      type=int)
parser.add_argument("--pool_stride",           type=int)
parser.add_argument("--nr_channels_1",         type=int, help="Number of channels in first layer")
parser.add_argument("--bottleneck_size",       type=int, help="Number of neurons in the bottleneck layer")

# Encodings
parser.add_argument("--encoding",              type=str, 
                    choices=["GAF","MTF","RP","SP","SC","GS"],
                    help="The encoding image matrix (e.g. X_gaf.npy) must be created first in Part 1!")


argp = parser.parse_args(
    ['--path_data','../Part1_Encoding',
     '--mode','new_training',
     '--dataset','training',
     '--cycles','50000',
     '--conv_kernel_size_1','4',
     '--conv_stride_1','2',
     '--pool_kernel_size','2',
     '--pool_stride','2',
     '--nr_channels_1','32', 
     '--bottleneck_size','160',
     '--batch_size','100',
     '--batch_size_testing','10',
     '--performance_eval_steps','10',
     '--checkpoint_save_steps','10000',
     '--encoding','GAF'])

### Import Encoding Matrix
########################################################

if argp.dataset=='training':
    
    if argp.encoding == 'GAF':
        X=np.load(os.path.join(argp.path_data,'GAF/GAF_Images/X_gaf.npy'))
    elif argp.encoding == 'MTF':
        X=np.load(os.path.join(argp.path_data,'MTF/MTF_Images/X_mtf.npy'))
    elif argp.encoding == 'RP':
        X=np.load(os.path.join(argp.path_data,'RP/RP_Images/X_rp.npy'))
    elif argp.encoding == 'SP':
        X=np.load(os.path.join(argp.path_data,'SP/SP_Images/X_sp.npy'))
    elif argp.encoding == 'SC':
        X=np.load(os.path.join(argp.path_data,'SC/SC_Images/X_sc.npy'))
    elif argp.encoding == 'GS':
        X=np.load(os.path.join(argp.path_data,'GS/GS_Images/X_gs.npy'))
    else:
        print('Check if encoding variable matches one of the following names:')
        print('GAF','MTF','RP','SP','SC','GS')

    n,l,_=X.shape
    print('number of images in training set:',n)
    
elif argp.dataset== 'testing': 
  
    if argp.encoding == 'GAF':
        X_test=np.load(os.path.join(argp.path_data,'GAF/GAF_Images/X_test_gaf.npy'))
    elif argp.encoding == 'MTF':
        X_test=np.load(os.path.join(argp.path_data,'MTF/MTF_Images/X_test_mtf.npy'))
    elif argp.encoding == 'RP':
        X_test=np.load(os.path.join(argp.path_data,'RP/RP_Images/X_test_rp.npy'))
    elif argp.encoding == 'SP':
        X_test=np.load(os.path.join(argp.path_data,'SP/SP_Images/X_test_sp.npy'))
    elif argp.encoding == 'SC':
        X_test=np.load(os.path.join(argp.path_data,'SC/SC_Images/X_test_sc.npy'))
    elif argp.encoding == 'GS':
        X_test=np.load(os.path.join(argp.path_data,'GS/GS_Images/X_test_gs.npy'))
    else:
        print('Check if encoding variable matches one of the following names:')
        print('GAF', 'MTF', 'RP', 'SP', 'SC', 'GS')

    n_valid,l,_=X_test.shape
    print('number of images in testing set:',n_valid)


print('Encoding:',argp.encoding)
print('image size:',l,'x',l)

class ConvolutionalAutoencoder(object):
    """
    Build the model using building Blocks
    
    """
    def __init__(self):
        """
        build the Graph
        """
        
        x = tf.placeholder(tf.float32, shape=[None, l, l, 1]) # [# batch, img_height, img_width, #channels]
        print('input',x.get_shape())
        
        h = Conv2D(filters=argp.nr_channels_1, 
                   kernel_size=argp.conv_kernel_size_1, 
                   strides=(argp.conv_stride_1, argp.conv_stride_1), 
                   padding='same',
                   activation=tf.nn.leaky_relu)(x)
        print('conv1',h.get_shape())
        
        pool1 = AveragePooling2D(pool_size=(argp.pool_kernel_size, argp.pool_kernel_size), 
                                 strides=argp.pool_stride, 
                                 padding='same')(h)
        print('pool1',pool1.get_shape())
        
        unfold = Unfold(scope='unfold')(pool1)
        print('unfold',unfold.get_shape())
        
        h = Dense(units=argp.bottleneck_size, activation=tf.nn.leaky_relu)(unfold)
        print('dense1',h.get_shape())
        
        h = Dense(units=int(unfold.get_shape()[1]), activation=tf.nn.leaky_relu)(h)
        print('dense2',h.get_shape())
        
        h = Fold(fold_shape = [-1, int(pool1.get_shape()[1]), int(pool1.get_shape()[1]), argp.nr_channels_1], 
                 scope      = 'fold')(h)
        print('fold',h.get_shape())
        
        h = UpSampling2D(size=(argp.pool_kernel_size,argp.pool_kernel_size))(h)
        print('unpool',h.get_shape())
        
        reconstruction = Conv2DTranspose(filters=1, 
                                         kernel_size=argp.conv_kernel_size_1, 
                                         strides=argp.conv_stride_1,
                                         padding='same',
                                         activation=tf.nn.leaky_relu)(h)
        
        print('reconstruction',reconstruction.get_shape())
        
        # Loss
        loss = tf.divide(tf.nn.l2_loss(x - reconstruction), argp.batch_size)  # L2 loss
        
        # Optimizer
        training = tf.train.AdamOptimizer(0.0001).minimize(loss)
        
        tf.summary.scalar('loss',loss)

        self.x = x
        self.reconstruction = reconstruction
        self.loss = loss
        self.training = training
        
    def train(self, batch_size, passes, new_training):
            """
            training process configuration

            """
            batch = Batch()

            with tf.Session() as sess:
                # prepare session
                if new_training:
                    print('new_training')

                    merged_summary = tf.summary.merge_all()
                    file_writer = tf.summary.FileWriter('tensorboard_summary', sess.graph)

                    saver, global_step = Model.start_new_session(sess)
                else:

                    merged_summary = tf.summary.merge_all()
                    file_writer = tf.summary.FileWriter('tensorboard_summary', sess.graph)

                    saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')

                # start training
                for step in range(1+global_step, 1+passes+global_step):
                    x = batch.get_batch(batch_size, dataset_name=argp.dataset, dataset=X)
                    self.training.run(feed_dict={self.x: x})

                    if step % argp.performance_eval_steps == 0:
                        print('Evaluating Performance...')
                        loss_eval = self.loss.eval(feed_dict={self.x: x})

                        result = sess.run(merged_summary,feed_dict={self.x: x})
                        file_writer.add_summary(result, step)

                        print("pass {}, training loss {}".format(step, loss_eval))

                    if step % argp.checkpoint_save_steps == 0:  # save weights
                        print('Saving Checkpoint...')
                        saver.save(sess, 'saver/cnn', global_step=step, write_meta_graph=True)
                        print('checkpoint saved')

    def reconstruct(self): 
        """
        reconstruction process configuration
        """

        batch = Batch()

        with tf.Session() as sess:
            saver, global_step = Model.continue_previous_session(sess, ckpt_file='saver/checkpoint')

            
            batch_size = argp.batch_size_testing

            if argp.dataset == 'training':
                x = batch.get_batch(batch_size, dataset_name=argp.dataset, dataset=X)
            elif argp.dataset == 'testing':
                x = batch.get_batch(batch_size, dataset_name=argp.dataset, dataset=X_test)
            else:
                sys.exit('Unexpected argument parser value for variable "dataset" ')

            org, recon = sess.run((self.x, self.reconstruction), feed_dict={self.x: x})

            org = np.squeeze(org,axis=3)
            recon = np.squeeze(recon,axis=3)
            
            
            for b in range (batch_size):
                fig_org, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
                plt.subplots_adjust(wspace=1)
                if argp.encoding!='GS':
                    image_org = axes[0].imshow(org[b],cmap='jet')
                    image_recon = axes[1].imshow(recon[b],cmap='jet')
                    image_residual = axes[2].imshow(np.abs(org[b]-recon[b]),cmap='jet')
                else:
                    image_org = axes[0].imshow(org[b],cmap='gray')
                    image_recon = axes[1].imshow(recon[b],cmap='gray')
                    image_residual = axes[2].imshow(np.abs(org[b]-recon[b]),cmap='gray')
                    
                # print(np.sum((org[b]-recon[b])**2)/2)
                plt.colorbar(image_org, fraction=0.0457, pad=0.04, ax=axes[0])
                plt.colorbar(image_recon, fraction=0.0457, pad=0.04, ax=axes[1])
                plt.colorbar(image_residual, fraction=0.0457, pad=0.04, ax=axes[2])
                
                # the colorbar limits can either be changed or left out entirely. In this case the limits get set automatically.
                if argp.encoding=='GAF':
                    image_org.set_clim(-1, 0.5)
                    image_recon.set_clim(-1, 0.5)
                    image_residual.set_clim(0,0.4)
                    axes[0].set_title('GAF: Original',fontsize=15)
                    axes[1].set_title('GAF: Reconstruction',fontsize=15)
                    axes[2].set_title('GAF: Residual',fontsize=15)
                  
                elif argp.encoding=='MTF':
                    image_org.set_clim(0, 0.4)
                    image_recon.set_clim(0,0.4)
                    image_residual.set_clim(0,0.1)
                    axes[0].set_title('MTF: Original',fontsize=15)
                    axes[1].set_title('MTF: Reconstruction',fontsize=15)
                    axes[2].set_title('MTF: Residual',fontsize=15)
                  
                elif argp.encoding=='RP':
                    image_org.set_clim(0, 2)
                    image_recon.set_clim(0, 2)
                    image_residual.set_clim(0, 1)
                    axes[0].set_title('RP: Original',fontsize=15)
                    axes[1].set_title('RP: Reconstruction',fontsize=15)
                    axes[2].set_title('RP: Residual',fontsize=15)
                  
                elif argp.encoding=='Spectro':
                    image_org.set_clim(0,-120)
                    image_recon.set_clim(0,-120)
                    image_residual.set_clim(0,40)
                    axes[0].set_title('SP: Original',fontsize=15)
                    axes[1].set_title('SP: Reconstruction',fontsize=15)
                    axes[2].set_title('SP: Residual',fontsize=15)
                  
                elif argp.encoding=='Scalo':
                    image_org.set_clim(np.min(org[b]*1.2),np.max(org[b]*1.2))
                    image_recon.set_clim(np.min(org[b]*1.2),np.max(org[b]*1.2))
#                   image_residual.set_clim(0, 0.5)
                    axes[0].set_title('SC: Original',fontsize=15)
                    axes[1].set_title('SC: Reconstruction',fontsize=15)
                    axes[2].set_title('SC: Residual',fontsize=15)
            
                elif argp.encoding=='GS':
                    image_org.set_clim(np.min(org[b]*1),np.max(org[b]*1))
                    image_recon.set_clim(np.min(org[b]*1),np.max(org[b]*1))
                    axes[0].set_title('GS: Original',fontsize=15)
                    axes[1].set_title('GS: Reconstruction',fontsize=15)
                    axes[2].set_title('GS: Residual',fontsize=15)

                plt.show()

def main():
    
    conv_autoencoder = ConvolutionalAutoencoder()
    
    if argp.mode == 'new_training':
        
        conv_autoencoder.train(batch_size   = argp.batch_size,
                               passes       = argp.cycles, 
                               new_training = True)
        
    elif argp.mode == 'continue_training' :

        conv_autoencoder.train(batch_size   = argp.batch_size,
                               passes       = argp.cycles, 
                               new_training = False)
    
    elif argp.mode == 'testing':
    
        conv_autoencoder.reconstruct()



if __name__ == '__main__':
    main()
