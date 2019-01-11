import tensorflow as tf
from sent_encoder import sent_encoder
from determiner import Determiner
from GatedCNNEncoder1 import CNNEncoder1

class main_model(tf.keras.Model):
    """
    desc. : This model passes each sentence of each paragraph to get a representation. then passess the sentence encodings
            of each paragraph through CNN and Dense layer afterwards
    """

    def __init__(self, name ,  embedding_table , hidden_len , filters=200,  encoder_dropout_rate=0.5 , cnn_dropout_rate=0.2 , classifier_dropout_rate = 0.2):
        """
        :param name: name of the model
        :param optimizer: optimizer!
        :param embedding_table: pre-trained embedding weights
        :param hidden_len: length of LSTM cell hidden
        :param encoder_dropout_rate: dropout rate of the encoder
        :param classifier_dropout_rate:  dropout rate of the classifier
        """
        super(main_model, self).__init__()
        self._name = name
        self._rnn_encoder = sent_encoder(self._name+'/LSTM_encoder', embedding_table, hidden_len)
        self._cnn_encoder = CNNEncoder1(self._name+'/GLU_CNN_encoder' , filters =filters)
        self._determiner = Determiner(name=self._name+'/FC_classifier')


    @property
    def variables(self):
        return self._encoder.variables+self._determiner.variables+self._cnn_encoder.variables


    def call(self,inputs):
        assert isinstance(inputs , list)

        first = inputs[2]
        sec = inputs[3]


        a = tf.range(tf.shape(first)[0])
        b = tf.expand_dims(a, axis=1)
        c = tf.tile(b, tf.constant([1, tf.shape(first)[1]]))
        range = tf.expand_dims(c, axis=2)

        first = tf.expand_dims(first, axis=2)
        sec = tf.expand_dims(sec, axis=2)


        first_indices = tf.concat((range, tf.dtypes.cast(first,tf.int32)), axis=2)
        sec_indices = tf.concat((range, tf.dtypes.cast(sec,tf.int32)), axis=2)

        x = tf.gather_nd(sent_encoder(inputs[0]),first_indices)
        x = self._cnn_encoder(x)

        y = tf.gather_nd(sent_encoder(inputs[1]),sec_indices)
        y = self._cnn_encoder(y)


        output = self._determiner(x , y)


        return output


