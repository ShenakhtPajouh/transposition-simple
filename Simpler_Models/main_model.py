import tensorflow as tf
from Simpler_Models.encoder import encoder
from Simpler_Models.determiner import Determiner


class main_model(tf.keras.Model):
    """
    desc. : this model encodes paragraphs using encoder (which is an LSTM here, but could be any other encoder), then concatenates the encodings and passes it
            through 2 Dense layers
    """

    def __init__(self, name , optimizer , embedding_table , hidden_len , encoder_dropout, classifier_dropout):
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
        self._optimizer = optimizer


    @property
    def variables(self):
        return self._encoder.variables+self._classifier.variables


    def call(self, inputs , targets):
        assert isinstance(inputs , list)

        with tf.GradientTape() as tape:
            x = encoder(inputs[0])
            y = encoder (inputs[1])
            output = self._classifier(tf.concat((x,y),axis=-1))
            loss = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=output))
            gradients = tape.gradient(loss, self.variables)
            self._optimizer.apply_gradients(zip(gradients, self.variables))
            return output

