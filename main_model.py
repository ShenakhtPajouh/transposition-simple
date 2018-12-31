import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import encoder
import classifier


class main_model(Model):
    """
    desc. : This model does the task of paragraph transposition. An abstract schema of this model would contain two
            input sequences, an encoder model and a classifier to determine the inputs order.
    """

    def __init__(self, name, **kwargs):
        super(main_model, self).__init__(**kwargs)
        self._name = name

        raise NotImplementedError

    def build(self, input_shape):
        raise NotImplementedError

    def call(self, inputs):
        raise NotImplementedError


