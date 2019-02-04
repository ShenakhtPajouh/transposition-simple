import tensorflow as tf
from tensorflow import keras
from staticRecurrentEntNet import simple_entity_network as entnet, EntityCell
import sys
sys.path.insert(0, '../Simpler_Models/')
from determiner import Determiner

class entity_main_model (keras.Model):
    def __init__(self, name, entity_num, entity_embedding_dim, rnn_hidden_size,max_sent_num , determiner_dropout_rate):
        super(entity_main_model, self).__init__()
        self._name=name
        self._entity_num = entity_num
        self._entity_embedding_dim = entity_embedding_dim
        self._rnn_hidden_size = rnn_hidden_size
        self._max_sent_num = max_sent_num
        self._entity_cell = EntityCell(max_entity_num=entity_num, entity_embedding_dim=entity_embedding_dim, name='entity_cell')
        self._determiner = Determiner(self._name+"/determiner", output_size=2, hidden_dim=64,
                                     dropout=determiner_dropout_rate)

    @property
    def variables(self):
        return self._determiner.variables + self._entity_cell.variables

    def call(self , inputs):
        """
        !!! here we concider the first sequence as concatenation of the first paragraph and the second paragraph
            and the second sequence as concatenation of the second paragraph and the first paragraph

        :param inputs: inputs[0] = a tensor of shape [batch_size, max_sent_num, max_token_num, embedding_len] containing the sequence
                       inputs[1] = a tensor of shape [batch_size , max_sent_num , max token_num] containing the mask of the sequence
                       inputs[2] = a tensor of shape [entity_num , entitiy_embedding_dim] containing embedding keys

        :return: a tensor of shape [batch_size , entity_num , 2] containing result of classification
        """

        x = entnet([inputs[0], inputs[1]], inputs[2], self._entity_cell)
        ret = 1 - self._determiner(x)
        return ret
