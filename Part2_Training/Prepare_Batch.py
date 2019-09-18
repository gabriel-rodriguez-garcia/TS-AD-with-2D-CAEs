### Prepare Batch #######################################################

from matplotlib import pyplot as plt
import numpy as np
from numpy import newaxis


class Batch(object):
    """
    Prepare data batches for training and testing.

    For Training the batches are selected randomly.

    """

    def __init__(self):
        self.training_images = None
        self.testing_images = None

    def get_batch(self, batch_size, dataset, dataset_name):
        """
        get a batch of images

        """

        if dataset_name == 'training':

            training_set_3d = dataset

            if self.training_images is None:
                self.training_images = training_set_3d
            #               # Pad images if necessary...
            #             images = np.pad(self.training_images, pad_width=[[0, 0], [3, 4],[3, 4]],mode= "constant",constant_values=0)

            images = self.training_images[:, :, :, newaxis]


        elif dataset_name == 'testing':
            validation_set_3d = dataset

            if self.testing_images is None:
                self.testing_images = validation_set_3d
            #               # Pad images if necessary...
            #             images = np.pad(self.training_images, pad_width=[[0, 0], [3, 4],[3, 4]],mode= "constant",constant_values=0)
            images = self.testing_images[:, :, :, newaxis]


        else:

            return

        num_samples = images.shape[0]
        idx = np.random.randint(num_samples, size=batch_size)

        return images[idx]