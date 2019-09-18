### Architecture SPECIAL Building Blocks
#######################################################

# Each Layer is defined as a class and later on used as a building block for the Architecture.
import tensorflow as tf
from abc import ABCMeta, abstractmethod


class Layer(object, metaclass=ABCMeta):
    """
    Abstract Class used for general Building Blocks

    """

    def __init__(self):
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class Unfold(Layer):
    """
    Unfold Building Block
    """

    def __init__(self,
                 scope=''):
        Layer.__init__(self)

        self.scope = scope

    def build(self, input_tensor):
        num_batch, height, width, num_channels = input_tensor.get_shape()

        return tf.reshape(input_tensor, [-1, (height * width * num_channels).value])

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)


class Fold(Layer):
    """
    Fold Building Block
    """

    def __init__(self,
                 fold_shape,
                 scope=''):
        Layer.__init__(self)

        self.fold_shape = fold_shape
        self.scope = scope

    def build(self, input_tensor):
        return tf.reshape(input_tensor, self.fold_shape)

    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)
