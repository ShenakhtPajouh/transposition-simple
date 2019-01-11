import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras.backend import repeat_elements, expand_dims
from tensorflow.keras.activations import softmax
import numpy as np
from determiner import Determiner

class AttentionBasedDeterminer2(Model):
    
    def __init__(self, name, par_len, dim, output_size = 1, dropout = 0.2):
        """
        This classifier determines if paragraphs are forward or backward
        Args:
            name: the name of the classifier
            par_len: number of sentences in each paragraph
            dim: dimension of input
            dropout: dropout value
        """
        
        super(AttentionBasedDeterminer2, self).__init__()
        self._name = name
        self.output_size = output_size
        self.dim = dim
        self.par_len = par_len
        self.dropout = layers.Dropout(dropout, name = name + '/Dropout')
        self.dense1 = layers.Dense(64, activation = 'relu', name = name + '/Dense1')
        self.dense2 = layers.Dense(1, name = name + '/Dense2')
    
    @property
    def variables(self):
        return [x.variables for x in self._layers]
    
    @property
    def trainable_variables(self):
        return [var for var in self._variables if self._trainable_variables[var]]

    def reshape(self, a, i, rep):
        a = expand_dims(a, axis = i)
        return repeat_elements(a, axis = i, rep = rep)
    
    def call(self, inputs, is_training = False):
        h, H, x, a = [], [], [], []
        h.append(inputs[0])
        H.append(inputs[1])
        h.append(inputs[2])
        H.append(inputs[3])
        
        for i in [0, 1]:
            H[i] = self.reshape(H[i], 1, self.par_len)
            h[i^1] = tf.convert_to_tensor(h[i^1], dtype=tf.float64)
            x = layers.concatenate([H[i], h[i^1]])
            a.append(self.dense1(x))
            if is_training:
                a[i] = self.dropout(a[i])
            a[i] = self.dense2(a[i])

        return a
