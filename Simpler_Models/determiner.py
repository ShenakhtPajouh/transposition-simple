import tensorflow as tf


class Determiner(tf.keras.Model):

    def __init__(self, name, num_classes=1, hidden_dim=64, dropout=0.2):
        """
        This classifier determines if paragraphs are forward or backward
        :param name: name of the classifier
        :param num_classes: dimension of output
        :param hidden_dim: dimension of hidden layer
        :param dropout: dropout value
        """

        super(Determiner, self).__init__()
        self._name = name
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer1 = tf.keras.layers.Dense(self.hidden_dim, activation='relu' , name=self._name+'/Dense1')
        self.layer2 = tf.keras.layers.Dense(self.num_classes, activation='tanh' , name=self._name+'/Dense2')

    @property
    def variables(self):
        return [x.variables for x in self._layers]

    @property
    def trainable_variables(self):
        return [var for var in self._variables if self._trainable_variables[var]]

    def call(self, inputs, is_training=True):
        y = self.layer1(inputs)
        if is_training:
            y = self.dropout(y)

        return self.layer2(y)