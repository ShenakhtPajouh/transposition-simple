import OS

print(OS.environ['PYTHONPATH'])

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from CNNEncoder1 import CNNEncoder1
from CNNEncoder2 import CNNEncoder2


class CNNEncoder3(Model):

    def __init__(self,
                 name,
                 batch_size,
                 paragraph_len,
                 sentence_len,
                 dim,
                 embedding_table,
                 hidden_len,
                 sent_num_layer=1,
                 par_num_layer=1,
                 kernel_size=3,
                 filters=1,
                 padding='SAME',
                 pool_size=3,
                 dropout=0.2):
        """
        Encoder of paragraph.
        Args:
            name: the name of the classifier
            embeding_table: the table of embeddings
            hidden_len: the dimensions of hidden layers
            num_layer: the number of layers in CNN
            kernel_size: the length of the 1D convolution window
            filters: the number of output filters in the convolution
            padding: value of padding
            pool_size: size of the max pooling windows
            dropout: the dropout value
        """

        super(CNNEncoder3, self).__init__()
        self._name = name

        self.model = Sequential()

        self.model.add(
            layers.Reshape(name=name + "/Reshape1",
                           target_shape=(batch_size * paragraph_len,
                                         sentence_len, dim)))

        self.model.add(
            CNNEncoder2(name=name + "/SentEncoder",
                        embedding_table=embedding_table,
                        hidden_len=hidden_len,
                        num_layer=sent_num_layer,
                        kernel_size=kernel_size,
                        filters=filters,
                        padding=padding,
                        pool_size=pool_size,
                        dropout=dropout))

        self.model.add(
            layers.Reshape(name=name + "/Reshape2",
                           target_shape=(batch_size, paragraph_len, dim)))

        self.model.add(
            CNNEncoder1(name=name + "/ParEncoder",
                        num_layer=par_num_layer,
                        kernel_size=kernel_size,
                        filters=filters,
                        padding=padding,
                        pool_size=pool_size,
                        dropout=dropout))

    def call(self, inputs, is_training=True):
        return self.model(inputs)
