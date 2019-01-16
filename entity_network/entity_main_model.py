import tensorflow as tf
from tensorflow import keras
from entity_network.staticRecurrentEntNet import StaticRecurrentEntNet
from Simpler_Models.determiner import Determiner


class entity_main_model (keras.Model):
    def __init__(self, name, entity_num, entity_embedding_dim, rnn_hidden_size,max_sent_num , determiner_dropout_rate):
        super(entity_main_model, self).__init__()
        self._name=name
        self._entity_num = entity_num
        self._entity_embedding_dim = entity_embedding_dim
        self._rnn_hidden_size = rnn_hidden_size
        self._max_sent_num = max_sent_num

        self._entnet = StaticRecurrentEntNet(self._entity_num, self._entity_embedding_dim,
                                             self._rnn_hidden_size,self._max_sent_num , name=self._name+"/entnet")
        self.determiner = Determiner(self._name+"/determiner", output_size=2, hidden_dim=64,
                                     dropout=determiner_dropout_rate)


    def call(self , inputs):
        """
        !!! here we concider the first sequence as concatenation of the first paragraph and the second paragraph
            and the second sequence as concatenation of the second paragraph and the first paragraph

        :param inputs: inputs[0] = a tensor of shape [batch_size,max_sent_num,max_token_num,embedding_len] containing the
                                   first sequence
                       inputs[1] = a tensor of shape [batch_size,max_sent_num,max_token_num,embedding_len] containing the
                                   second sequence
                       inputs[2] = a tensor of shape [batch_size , max_sent_num , max token_num] containing the mask of
                                   first sequence
                       inputs[3] = a tensor of shape [batch_size , max_sent_num , max token_num] containing the mask of
                                   first sequence
                       inputs[4] = a tensor of shape [entity_num , entitiy_embedding_dim] containing embedding keys

        :return: a tensor of shape [batch_size , 2] containing result of classification
        """

        x = self._entnet([inputs[4], inputs[0], inputs[2]])
        y = self._entnet ([inputs[4],inputs[1],inputs[3]])

        ret = self._determiner(tf.concat((x,y),axis=-1))

