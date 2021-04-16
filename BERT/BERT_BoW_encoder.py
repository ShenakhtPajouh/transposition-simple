import tensorflow as tf


class encoder(tf.keras.Model):

    def __init__(self, name):
        super(encoder, self).__init__()

        self._name = name

    def call(self, seq):
        """
        desc. : here we use extracted features of sentence tokens and use their mean as the representation of sentence
        :param seq: containing
        :return: encoding of each sequence which is mean of embeddings of its tokens
        """

        output = tf.reduce_mean(seq, -1, keep_dims=True)

        return output
