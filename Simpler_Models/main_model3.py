import tensorflow as tf
from Simpler_Models.sent_encoder import sent_encoder
from Simpler_Models.determiner import Determiner


class main_model(tf.keras.Model):
    """
    desc. : This model passes each sentence of each paragraph to get a representation. then passess the sentence encodings
            of each paragraph through CNN and Dense layer afterwards
    """

    def __init__(self, name , optimizer , embedding_table , hidden_len , encoder_dropout_rate , classifier_dropout_rate):
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
        self._encoder = sent_encoder(self._name+'/LSTM_encoder', embedding_table, hidden_len, encoder_dropout_rate)
        self._determiner = Determiner(name=self._name+'/FC_classifier' , dropout=classifier_dropout_rate)
        self._optimizer = optimizer


    @property
    def variables(self):
        return self._encoder.variables+self._classifier.variables


    def call(self,inputs , indices, targets):
        assert isinstance(inputs , list)

        with tf.GradientTape() as tape:
            x = tf.gather_nd(sent_encoder(inputs[0]),indices[0])
            y = tf.gather_nd(sent_encoder(inputs[1]),indices[1])
            output = self._determiner(x , y)
            loss = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=output))
            gradients = tape.gradient(loss, self.variables)
            self._optimizer.apply_gradients(zip(gradients, self.variables))
            return output


