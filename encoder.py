import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import *


class encoder(Model):
    """
    desc. : Used for encoding a sequence of words. It does not necessarily return embedding, but may return a list of
            hidden states or ... .
    """

    def __init__(self, name, **kwargs):
        super(encoder, self).__init__(**kwargs)
        self._name = name

        raise NotImplementedError

    def build(self, input_shape):
        raise NotImplementedError

    def call(self, inputs):
        raise NotImplementedError
