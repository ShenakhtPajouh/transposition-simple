import tensorflow as tf
import numpy as np


class Determiner(tf.keras.Model):

    def __init__(self, name, output_size=2, hidden_dim=64, dropout=0.2):
        """
        This classifier determines if paragraphs are forward or backward
        Args:
            name: the name of the classifier
            num_classes: dimension of output
            hidden_dim: dimension of hidden layer
            dropout: dropout value
        """

        super(Determiner, self).__init__()
        self._name = name
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.dropout = tf.keras.layers.Dropout(dropout, name=name + '/Dropout')
        self.layer1 = tf.keras.layers.Dense(self.hidden_dim,
                                            activation='relu',
                                            name=name + '/Dense1')
        self.layer2 = tf.keras.layers.Dense(self.output_size,
                                            activation='softmax',
                                            name=name + '/Dense2')

    @property
    def variables(self):
        return [x.variables for x in self._layers]

    @property
    def trainable_variables(self):
        return [
            var for var in self._variables if self._trainable_variables[var]
        ]

    def call(self, inputs, is_training=True):
        y = self.layer1(inputs)
        if is_training:
            y = self.dropout(y)
        return self.layer2(y)
