import tensorflow as tf
from tensorflow.keras.layers import *

class sent_encoder(tf.keras.Model):
    def __init__(self, name, embedding_table , hidden_len, dropout_rate):
        super(sent_encoder, self).__init__()

        self._name = name
        self._vocab_size = embedding_table.shape[0]
        self._embedding_len = embedding_table.shape[1]
        self._hidden_len = hidden_len
        self.dropout_rate = dropout_rate


        embedding_initializer = tf.keras.initializers.constant(embedding_table)
        self._embeddings = tf.keras.layers.Embedding(self._vocab_size , self._embedding_len , embeddings_initializer= embedding_initializer ,
                                     trainable=False , mask_zero=True , name=self._name+'/embedding')
        self._LSTM = tf.keras.layers.LSTM(self._hidden_len , name=self._name+'/LSTM' , return_sequences=True)


    @property
    def variables(self):
        return self._LSTM.variables


    def call(self , seq , ):
        x = self._embeddings(seq)
        outputs = self._LSTM(x)

        return outputs

