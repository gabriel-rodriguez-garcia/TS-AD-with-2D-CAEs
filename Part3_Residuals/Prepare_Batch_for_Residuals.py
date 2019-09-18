### Prepare Batch for Residuals
########################################################
import numpy as np

class Batch(object):
    """
    Prepare data batches for reconstruction
    """

    def __init__(self):
        self.training_images = None
        self.testing_images = None

    def get_batch(self, dataset, dataset_name, part):
        """
        get a batch of images.

        """
        n=np.shape(dataset)[0]

        if dataset_name == 'training':

            if part == 'part 1':
                idx = np.linspace(0, n // 2 - 1, num=n // 2, dtype=int)
            elif part == 'part 2':
                idx = np.linspace(n // 2, n - 1, num=n // 2, dtype=int)
            else:
                print('choose either part 1 or part 2 of the training dataset to generate')

            return dataset[idx, :, :, np.newaxis]

        elif dataset_name == 'testing':

            idx = np.linspace(0, n - 1, num=n, dtype=int)
            return dataset[idx, :, :, np.newaxis]

        else:
            return