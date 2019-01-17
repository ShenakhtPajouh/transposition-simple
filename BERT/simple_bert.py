import tensorflow as tf
from tensorflow import keras
from Simpler_Models.determiner import Determiner


class simple_bert(keras.Model):
    def __init__(self , name , determiner_dropout_rate=0.2):
        super(simple_bert, self).__init__()
        self._name=name
        self.__determiner = Determiner(self._name+"/determiner", output_size=2, hidden_dim=64,
                                     dropout=determiner_dropout_rate)

    @property
    def variables(self):
        return self._determiner.variables

    def call(self ,input):
        """
        :param inputs: bert encoding of concatenation of last part of first paragraph and first part of the second
                       paragraph (with size at most 512)
        """

        return self.__determiner(input)
