import numpy as np
import tensorflow as tf
import argparse
import os
import sys

from keras.models import Model
from keras.layers import Dense,Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, UpSampling2D

from Part3_Residuals.Prepare_Batch_for_Residuals import Batch
from Part2_Training.Special_Layers import Fold, Unfold
from Part2_Training.Training_Schedule import Model

### Global Parameters
########################################################

# Change Parameters here and use them everywhere in the code using: argp.<variable_name>

parser = argparse.ArgumentParser()

# Paths/Directories
parser.add_argument("--path_data", type=str, help="Path to folder of input data")

# Work Schedule
parser.add_argument("--mode", choices=["testing"])
parser.add_argument("--dataset", choices=["training","testing"])
parser.add_argument("--cycles",                type=int)
parser.add_argument("--performance_eval_steps",type=int, help="Interval: number of steps to compute loss")
parser.add_argument("--checkpoint_save_steps", type=int, help="Interval: number of steps to a checkpoint of the Model state")
parser.add_argument("--batch_size_testing",    type=int, help="Number of images to show when testing")
parser.add_argument("--batch_size",            type=int, help="Number of images to use for computing loss while training")
parser.add_argument("--part", choices=["part 1","part 2"], help ="Which part of the training dataset to reconstruct, the first or the second half")# This is necessary because kernel memory issues


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
                    help="Note, the encoding image matrix (e.g. X_gaf_red.npy) must be saved in the Dataset folder!")


argp = parser.parse_args(
    ['--path_data','../Part1_Encoding',
     '--mode','testing',
     '--dataset','testing',
     '--part','part 2',
     '--cycles','50000',
     '--conv_kernel_size_1','4',
     '--conv_stride_1','2',
     '--pool_kernel_size','2',
     '--pool_stride','2',
     '--nr_channels_1','32',
     '--bottleneck_size','160',
     '--batch_size','100',
     '--batch_size_testing','20',
     '--performance_eval_steps','10',
     '--checkpoint_save_steps','10000',
     '--encoding','GS'])


### Import Encoding Matrix
########################################################

if argp.dataset == 'training':

    if argp.encoding == 'GAF':
        X = np.load(os.path.join(argp.path_data, 'GAF/GAF_Images/X_gaf.npy'))
    elif argp.encoding == 'MTF':
        X = np.load(os.path.join(argp.path_data, 'MTF/MTF_Images/X_mtf.npy'))
    elif argp.encoding == 'RP':
        X = np.load(os.path.join(argp.path_data, 'RP/RP_Images/X_rp.npy'))
    elif argp.encoding == 'SP':
        X = np.load(os.path.join(argp.path_data, 'SP/SP_Images/X_sp.npy'))
    elif argp.encoding == 'SC':
        X = np.load(os.path.join(argp.path_data, 'SC/SC_Images/X_sc.npy'))
    elif argp.encoding == 'GS':
        X = np.load(os.path.join(argp.path_data, 'GS/GS_Images/X_gs.npy'))
    else:
        print('Check if encoding variable matches one of the following names:')
        print('GAF', 'MTF', 'RP', 'SP', 'SC', 'GS')

    n, l, _ = X.shape
    print('number of images in training set:', n)

elif argp.dataset == 'testing':

    if argp.encoding == 'GAF':
        X_test = np.load(os.path.join(argp.path_data, 'GAF/GAF_Images/X_test_gaf.npy'))
    elif argp.encoding == 'MTF':
        X_test = np.load(os.path.join(argp.path_data, 'MTF/MTF_Images/X_test_mtf.npy'))
    elif argp.encoding == 'RP':
        X_test = np.load(os.path.join(argp.path_data, 'RP/RP_Images/X_test_rp.npy'))
    elif argp.encoding == 'SP':
        X_test = np.load(os.path.join(argp.path_data, 'SP/SP_Images/X_test_sp.npy'))
    elif argp.encoding == 'SC':
        X_test = np.load(os.path.join(argp.path_data, 'SC/SC_Images/X_test_sc.npy'))
    elif argp.encoding == 'GS':
        X_test = np.load(os.path.join(argp.path_data, 'GS/GS_Images/X_test_gs.npy'))
    else:
        print('Check if encoding variable matches one of the following names:')
        print('GAF', 'MTF', 'RP', 'SP', 'SC', 'GS')

    n_valid, l, _ = X_test.shape
    print('number of images in testing set:', n_valid)

print('Encoding:', argp.encoding)
print('image size:', l, 'x', l)


### Network and Testing
########################################################

class ConvolutionalAutoencoder(object):
    """
    Build the model using building Blocks
    
    """
    def __init__(self):
        """
        build the Graph
        """

        x = tf.placeholder(tf.float32, shape=[None, l, l, 1])  # [# batch, img_height, img_width, #channels]
        print('input', x.get_shape())

        h = Conv2D(filters=argp.nr_channels_1,
                   kernel_size=argp.conv_kernel_size_1,
                   strides=(argp.conv_stride_1, argp.conv_stride_1),
                   padding='same',
                   activation=tf.nn.leaky_relu)(x)
        print('conv1', h.get_shape())

        pool1 = AveragePooling2D(pool_size=(argp.pool_kernel_size, argp.pool_kernel_size),
                                 strides=argp.pool_stride,
                                 padding='same')(h)
        print('pool1', pool1.get_shape())

        unfold = Unfold(scope='unfold')(pool1)
        print('unfold', unfold.get_shape())

        h = Dense(units=argp.bottleneck_size, activation=tf.nn.leaky_relu)(unfold)
        print('dense1', h.get_shape())

        h = Dense(units=int(unfold.get_shape()[1]), activation=tf.nn.leaky_relu)(h)
        print('dense2', h.get_shape())

        h = Fold(fold_shape=[-1, int(pool1.get_shape()[1]), int(pool1.get_shape()[1]), argp.nr_channels_1],
                 scope='fold')(h)
        print('fold', h.get_shape())

        h = UpSampling2D(size=(argp.pool_kernel_size, argp.pool_kernel_size))(h)
        print('unpool', h.get_shape())

        reconstruction = Conv2DTranspose(filters=1,
                                         kernel_size=argp.conv_kernel_size_1,
                                         strides=argp.conv_stride_1,
                                         padding='same',
                                         activation=tf.nn.leaky_relu)(h)

        print('reconstruction', reconstruction.get_shape())

        # Loss
        loss = tf.divide(tf.nn.l2_loss(x - reconstruction), argp.batch_size)  # L2 loss

        # Optimizer
        training = tf.train.AdamOptimizer(0.0001).minimize(loss)

        tf.summary.scalar('loss', loss)

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
                file_writer = tf.summary.FileWriter('../Part2_Training/tensorboard_summary', sess.graph)
                
                saver, global_step = Model.start_new_session(sess)
            else:
                
                merged_summary = tf.summary.merge_all()
                file_writer = tf.summary.FileWriter('../Part2_Training/tensorboard_summary', sess.graph)
                
                saver, global_step = Model.continue_previous_session(sess, ckpt_file='../Part2_Training/saver/checkpoint')

            # start training
            for step in range(1+global_step, 1+passes+global_step):
                x = batch.get_batch(dataset_name=argp.dataset,dataset=X,part=argp.part)
                self.training.run(feed_dict={self.x: x})

                if step % argp.performance_eval_steps == 0:
                    print('Evaluating Performance...')
                    loss_eval = self.loss.eval(feed_dict={self.x: x})
                    
                    result = sess.run(merged_summary,feed_dict={self.x: x})
                    file_writer.add_summary(result, step)
                    
                    print("pass {}, training loss {}".format(step, loss_eval))

                if step % argp.checkpoint_save_steps == 0:  # save weights
                    print('Saving Checkpoint...')
                    saver.save(sess, '../Part2_Training/saver/cnn', global_step=step, write_meta_graph=True)
                    print('checkpoint saved')
                    

    def reconstruct(self): 
                
        batch = Batch()

        with tf.Session() as sess:
            saver, global_step = Model.continue_previous_session(sess, ckpt_file='../Part2_Training/saver/checkpoint')
            
            batch_size = 1

            if argp.dataset == 'training':
                x = batch.get_batch( dataset_name=argp.dataset, dataset=X, part=argp.part)
            elif argp.dataset == 'testing':
                x = batch.get_batch(dataset_name=argp.dataset, dataset=X_test,part=argp.part)
            else:
                sys.exit('Unexpected argument parser value for variable "dataset" ')

            if argp.dataset=='training':

                if argp.encoding == 'GS':
                    k=1677
                else:
                    k=2340

                errors=np.zeros((n//2,1,1))

                for i in range(0,k):
                    print(i)
                    org, recon = sess.run((self.x, self.reconstruction), feed_dict={self.x: x[n//k*i:n//k*(i+1),:,:]})
                    errors[n//k*i:n//k*(i+1),0,0]=np.sum(np.abs(np.squeeze(org,axis=3)-np.squeeze(recon,axis=3)),axis=(1,2))

                np.save('train_errors_'+argp.part+'.npy',errors)
                
            elif argp.dataset=='testing':
                
                errors=np.zeros((n_valid,1,1))

                if argp.encoding=='GS':
                    k=108
                else:
                    k=120

                for i in range(0,k):
                    print(i)
                    org, recon = sess.run((self.x, self.reconstruction), feed_dict={self.x: x[n_valid//k*i:n_valid//k*(i+1),:,:]})
                    errors[n_valid//k*i:n_valid//k*(i+1),0,0]=np.sum(np.abs(np.squeeze(org,axis=3)-np.squeeze(recon,axis=3)),axis=(1,2))

                np.save('test_errors.npy',errors)

def main():
    
    conv_autoencoder = ConvolutionalAutoencoder()
    
    if argp.mode == 'testing':
    
        conv_autoencoder.reconstruct()
        
    else:
        print('Change mode to testing!')


if __name__ == '__main__':
    main()
