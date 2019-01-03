import tensorflow as tf
from encoder import encoder
from determiner import Determiner


class main_model(tf.keras.Model):
    """
    desc. : this model encodes paragraphs using encoder (which is an LSTM here, but could be any other encoder), then concatenates the encodings and passes it
            through 2 Dense layers
    """

    def __init__(self, name  , embedding_table , hidden_len , encoder_dropout=0.5, classifier_dropout=0.2):
        """
        :param name: name of the model
        :param optimizer: optimizer!
        :param embedding_table: pre-trained embedding weights
        :param hidden_len: length of LSTM cell hidden
        :param encoder_dropout: dropout rate of the encoder
        :param classifier_dropout:  dropout rate of the classifier
        """
        super(main_model, self).__init__()
        self._name = name
        self._encoder = encoder(self._name+'/LSTM_encoder', embedding_table, hidden_len, encoder_dropout)
        self._determiner = Determiner(name=self._name+'/FC_classifier' , dropout=classifier_dropout)


    @property
    def variables(self):
        return self._encoder.variables+self._determiner.variables


    def call(self, inputs):
        assert isinstance(inputs , list)

        x = self._encoder(inputs[0])
        y = self._encoder (inputs[1])
        output = self._determiner(tf.concat((x,y),axis=-1))
        return output


