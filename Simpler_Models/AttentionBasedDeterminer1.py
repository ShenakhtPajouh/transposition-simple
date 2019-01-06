import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras.backend import repeat_elements, expand_dims
from tensorflow.keras.activations import softmax
import numpy as np
from determiner import Determiner

class AttentionBasedClassifier1(Model):

    def __init__(self, name, par_len, dim, output_size=1, dropout=0.2):
        """
        This classifier determines if paragraphs are forward or backward
        Args:
            name: the name of the classifier
            par_len: number of sentences in each paragraph
            dim: dimension of input
            dropout: dropout value
        """

        super(AttentionBasedClassifier1, self).__init__()
        self._name = name
        self.output_size = output_size
        self.dim = dim
        self.par_len = par_len
        self.dropout = layers.Dropout(dropout, name=name + '/Dropout')
        self.dense = layers.Dense(1, name=name + '/Dense')
        self.FC = Determiner(name=name + "/FC")

    @property
    def variables(self):
        return [x.variables for x in self._layers]

    @property
    def trainable_variables(self):
        return [var for var in self._variables if self._trainable_variables[var]]

    def reshape(self, a, i, rep):
        a = expand_dims(a, axis=i)
        return repeat_elements(a, axis=i, rep=rep)

    def call(self, inputs, is_training=False):
        h = inputs[0]
        q = inputs[1]
        dim = 2 * self.dim
        x = layers.concatenate([self.reshape(h, 2, self.par_len), self.reshape(q, 2, self.par_len)])
        x = tf.reshape(x, shape=(x.shape[0], self.par_len * self.par_len, dim))
        a = self.dense(x)
        if is_training:
            a = self.dropout(a)
        a = tf.reshape(a, shape=(a.shape[0], self.par_len * self.par_len))
        a = softmax(a, axis=1)
        a = self.reshape(a, 2, dim)
        y = tf.reduce_sum(a * x, axis=1)
        return self.FC(y)