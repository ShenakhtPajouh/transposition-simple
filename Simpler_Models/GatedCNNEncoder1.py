import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np


class CNNEncoder1(Model):

    def __init__(self,
                 name,
                 num_layer=1,
                 kernel_size=5,
                 filters=1,
                 padding='SAME',
                 dropout=0.2):
        """
        Encoder of paragraph.
        Args:
            name: the name of the classifier
            num_layer: the number of layers in CNN
            kernel_size: the length of the 1D convolution window
            filters: the number of output filters in the convolution
            padding: value of padding
            pool_size: size of the max pooling windows
            dropout: the dropout value
        """

        super(CNNEncoder1, self).__init__()
        self._name = name
        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.filters = filters
        self.dropout = layers.Dropout(dropout)
        self.conv = layers.Conv1D(kernel_size=kernel_size,
                                  filters=filters,
                                  padding=padding,
                                  name=name + '/Conv1D')
        self.gate = layers.Conv1D(kernel_size=kernel_size,
                                  filters=filters,
                                  padding=padding,
                                  activation="sigmoid",
                                  name=name + '/Gate')
        self.global_pool = layers.GlobalAveragePooling1D(name=name +
                                                         '/GlobalAvePool1D')

    @property
    def variables(self):
        return [x.variables for x in self._layers]

    @property
    def trainable_variables(self):
        return [
            var for var in self._variables if self._trainable_variables[var]
        ]

    def call(self, inputs, is_training=True):
        y = inputs

        for i in range(0, self.num_layer):
            A = self.conv(y)
            B = self.gate(y)
            y = layers.Multiply()([A, B])
            if is_training:
                y = self.dropout(y)

        return self.global_pool(y)
