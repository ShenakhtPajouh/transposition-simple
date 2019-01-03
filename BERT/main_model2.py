import tensorflow as tf
from brt.BERT_BoW_encoder import encoder
from Simpler_Models.determiner import Determiner

class main_model(tf.keras.Model):
    """
    desc. : This model does the task of paragraph transposition. An abstract schema of this model would contain two
            input sequences, an encoder model and a classifier to determine the inputs order.
    """

    def __init__(self, name , optimizer , classifier_dropout_rate):
        """
        :param name: name of the model
        :param optimizer: optimizer!
        :param classifier_dropout_rate:  dropout rate of the classifier
        """

        super(main_model, self).__init__()
        self._name = name
        self._encoder = encoder(self._name+'/BoW_encoder')
        self._determiner = Determiner(name=self._name+'/FC_classifier' , dropout=classifier_dropout_rate)
        self._optimizer = optimizer


    @property
    def variables(self):
        return self._classifier.variables

    @property
    def trainable_variables(self):
        return self._classifier.trainable_variables


    def call(self, inputs , targets):
        assert isinstance(inputs , list)

        with tf.GradientTape() as tape:
            x = encoder(inputs[0])
            y = encoder (inputs[1])
            output = self._classifier(x , y)
            loss = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=output))
            gradients = tape.gradient(loss, self.variables)
            self._optimizer.apply_gradients(zip(gradients, self.variables))
            return output



