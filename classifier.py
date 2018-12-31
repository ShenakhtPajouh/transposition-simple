import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import *


class classifier(Model):
    """
    desc. : Used for the last stage of our model. using inputs (which may be two encodings, or two sequence of
            hidden states or ...), this model determines which one has occurred first.
    """

    def __init__(self, name, **kwargs):
        super(classifier, self).__init__(**kwargs)
        self._name = name

        raise NotImplementedError

    def build(self, input_shape):
        raise NotImplementedError

    def call(self, inputs):
        raise NotImplementedError