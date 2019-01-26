import tensorflow as tf
from tensorflow.contrib import autograph
import standard.prgrph_ending_classifier as prgrph_ending_classifier

K = tf.keras.backend


class Sent_encoder(tf.keras.Model):
    def __init__(self, name=None):
        if name is None:
            name = 'sent_encoder'
        super().__init__(name=name)

    def call(self, inputs):
        """
        Description:
            encode given sentences with bag of words algorithm
        Args:
            input: sents shape: [current_prgrphs_num,max_sent_len,embedding_dim]
            output: encoded sentences of shape [current_prgrphs_num,encoding_dim] , here encoding_dim is equal to embedding_dim
        """
        ' I assume word embedding for indexes greater that sentnece length is zero vector, so it does not effect sentence encoding '

        return tf.reduce_sum(inputs, 1)


class EntityCell(tf.keras.Model):
    """
    Entity Cell.
    call with inputs and keys
    """

    def __init__(self, max_entity_num, entity_embedding_dim, activation=tf.nn.relu, name=None, initializer=None,
                 **kwargs):
        if name is None:
            name = 'Entity_cell'
        super().__init__(name=name)
        self.max_entity_num = max_entity_num
        self.entity_embedding_dim = entity_embedding_dim
        self.activation = activation
        if initializer is None:
            self.initializer = tf.keras.initializers.random_normal()

        self.U = None
        self.V = None
        self.W = None
        self.built = False

    def build(self, input_shape):
        shape = [self.entity_embedding_dim, self.entity_embedding_dim]
        self.U = K.variable(self.initializer(shape), name='U')
        self.V = K.variable(self.initializer(shape), name='V')
        self.W = K.variable(self.initializer(shape), name='W')
        self.built = True

    def get_gate(self, encoded_sents, current_hiddens, current_keys):
        """
        Description:
            calculate the gate g_i for all hiddens of given paragraphs
        Args:
            inputs: encoded_sents of shape: [current_prgrphs_num, encoding_dim]
                    current_hiddens: [current_prgrphs_num, entity_num, entity_embedding_dim]
                    current_keys: [current_prgrphs_num, entity_num, entity_embedding_dim]

            output: gates of shape : [curr_prgrphs_num, entity_num]
        """

        print('enocded_sents dtype:', encoded_sents.dtype)
        print('current_hiddens dtype:', current_hiddens.dtype)
        print('enocded_sents shape:', encoded_sents.shape)
        return tf.sigmoid(tf.reduce_sum(tf.multiply(tf.expand_dims(encoded_sents, 1), current_hiddens) +
                                        tf.multiply(tf.expand_dims(encoded_sents, 1), current_keys), axis=2))

    def update_hidden(self, gates, current_hiddens, current_keys, encoded_sents):
        """
        Description:
            updates hidden_index for all prgrphs
        Args:
            inputs: gates shape: [current_prgrphs_num, entity_num]
                    encoded_sents of shape: [current_prgrphs_num, encoding_dim]
                    current_hiddens: [current_prgrphs_num, entity_num, entity_embedding_dim]
                    current_keys: [current_prgrphs_num, entity_num, entity_embedding_dim]
        """
        curr_prgrphs_num = current_hiddens.shape[0]
        h_tilda = self.activation(
            tf.reshape(tf.matmul(tf.reshape(current_hiddens, [-1, self.entity_embedding_dim]), self.U) +
                       tf.matmul(tf.reshape(current_keys, [-1, self.entity_embedding_dim]), self.V) +
                       tf.matmul(tf.reshape(tf.tile(tf.expand_dims(encoded_sents, 1), [1, self.max_entity_num, 1]),
                                            shape=[-1, self.entity_embedding_dim]), self.W),
                       shape=[curr_prgrphs_num, self.max_entity_num, self.entity_embedding_dim]))
        'h_tilda shape: [current_prgrphs_num, entity_num, entity_embedding_dim]'
        # tf.multiply(gates,h_tilda)
        updated_hiddens = current_hiddens + tf.multiply(
            tf.tile(tf.expand_dims(gates, axis=2), [1, 1, self.entity_embedding_dim]), h_tilda)

        return updated_hiddens

    def normalize(self, hiddens):
        return tf.nn.l2_normalize(hiddens, axis=2)

    def call(self, inputs, prev_states, keys, use_shared_keys=False, **kwargs):
        """

        Args:
            inputs: encoded_sents of shape [batch_size, encoding_dim] , batch_size is equal to current paragraphs num
            prev_states: tensor of shape [batch_size, key_num, dim]
            keys: tensor of shape [batch_size, key_num, dim] if use_shared_keys is False and
                  [key_num, dim] if use_shared_keys is True
            use_shared_keys: if it is True, it use shared keys for all samples.

        Returns:
            next_state: tensor of shape [batch_size, key_num, dim]
        """

        # assert isinstance(inputs, list)

        # current_hiddens = tf.gather(self.hiddens, indices)
        # print('ENCODE')
        # print(self.keys.shape)
        # print(indices)
        # current_keys = tf.gather(self.keys, indices)

        # if current_hiddens.shape != current_keys.shape:
        #     raise AttributeError('hiddens and kes must have same shape')

        encoded_sents = inputs
        gates = self.get_gate(encoded_sents, prev_states, keys)
        updated_hiddens = self.update_hidden(gates, prev_states, keys, encoded_sents)
        return self.normalize(updated_hiddens)

    def get_initial_state(self):
        return tf.zeros([self.max_entity_num, self.entity_embedding_dim], dtype=tf.float32)

    # def __call__(self, inputs, prev_state, keys, use_shared_keys=False, **kwargs):
    #     """
    #     Do not fill this one
    #     """
    #     return super().__call__(inputs=inputs, prev_state=prev_state, keys=keys,
    #                             use_shared_keys=use_shared_keys, **kwargs)


# @autograph.convert()
def simple_entity_network(inputs, keys, entity_cell=None,
                          initial_entity_hidden_state=None,
                          use_shared_keys=False, return_last=True):
    """
    Args:
        entity_cell: the EntityCell
        inputs: a list containing a tensor of shape [batch_size, seq_length, dim] and its mask of shape [batch_size, seq_length]
                batch_size=current paragraphs num, seq_length=max number of senteces
        keys: tensor of shape [batch_size, key_num, dim] if use_shared_keys is False and
                  [key_num, dim] if use_shared_keys is True
        use_shared_keys: if it is True, it use shared keys for all samples.
        mask_inputs: tensor of shape [batch_size, seq_length] and type tf.bool
        initial_entity_hidden_state: a tensor of shape [batch_size, key_num, dim]
        return_last: if it is True, it returns the last state, else returns all states

    Returns:
        if return_last = True then a tensor of shape [batch_size, key_num, dim] else shape of
                         [batch_size, seq_length, key_num, dim]
    """

    encoded_sents, mask = inputs
    seq_length = encoded_sents.shape[1]
    batch_size = encoded_sents.shape[0]
    key_num = keys.shape[1]
    entity_embedding_dim = keys.shape[2]

    if entity_cell is None:
        entity_cell = EntityCell(max_entity_num=key_num, entity_embedding_dim=entity_embedding_dim,
                                 name='entity_cell')

    if initial_entity_hidden_state is None:
        initial_entity_hidden_state = tf.tile(tf.expand_dims(entity_cell.get_initial_state(), axis=0),
                                              [batch_size, 1, 1])
    if return_last:
        entity_hiddens = initial_entity_hidden_state
    else:
        all_entity_hiddens = tf.expand_dims(initial_entity_hidden_state, axis=1)

    for i in range(seq_length):
        ''' to see which sentences are available '''
        indices = tf.where(mask[:, i])
        indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
        curr_encoded_sents = tf.gather(encoded_sents[:, i, :], indices)
        curr_keys = tf.gather(keys, indices)
        if return_last:
            prev_states = tf.gather(entity_hiddens, indices)
            updated_hiddens = entity_cell(curr_encoded_sents, prev_states, curr_keys)
            entity_hiddens = entity_hiddens + tf.scatter_nd(tf.expand_dims(indices, 1), updated_hiddens - prev_states,
                                                            keys.shape)
        else:
            prev_states = tf.gather(all_entity_hiddens[:, -1, :, :], indices)
            updated_hiddens = tf.expand_dims(entity_cell(curr_encoded_sents, prev_states, curr_keys), axis=1)
            all_entity_hiddens = tf.concat([all_entity_hiddens,
                                            tf.scatter_nd(tf.expand_dims(indices, 1), updated_hiddens,
                                                          [batch_size, 1, key_num, entity_embedding_dim])], axis=1)

    if return_last:
        return entity_hiddens
    else:
        return all_entity_hiddens


@autograph.convert()
def rnn_entity_network_encoder(entity_cell, rnn_cell, inputs, keys, mask_inputs=None,
                               initial_hidden_state=None,
                               initial_entity_hidden_state=None, update_positions=None, use_shared_keys=False,
                               return_last=True, self_attention=False):
    """


    """
    raise NotImplementedError


@autograph.convert()
def rnn_entity_network_decoder(entity_cell, rnn_cell, softmax_layer, embedding_layer, keys, training,
                               initial_hidden_state=None, initial_entity_hidden_state=None,
                               labels=None,
                               num_inputs=None, num_keys=None, encoder_hidden_states=None,
                               update_positions=None, use_shared_keys=False, return_last=True,
                               attenton=False, self_attention=False):
    """

    Args:
        entity_cell: EntityCell
        rnn_cell: RNNCell
        softmax_layer: softmax layer
        embedding_layer: embedding layer
        keys: either a tensor of shape [batch_size, key_num, dim] or [key_num, dim] depending on use_shared_keys
        training: boolean ...
    :param initial_hidden_state:
    :param initial_entity_hidden_state:
    :param labels:
    :param num_inputs:
    :param num_keys:
    :param encoder_hidden_states:
    :param update_positions:
    :param use_shared_keys:
    :param return_last:
    :param attenton:
    :param self_attention:
    :return:
    """
    raise NotImplementedError


"""

"""


class BasicRecurrentEntityEncoder(tf.keras.Model):
    def __init__(self, embedding_matrix, max_entity_num=None, entity_embedding_dim=None, entity_cell=None, name=None,
                 **kwargs):
        if name is None:
            name = 'BasicRecurrentEntityEncoder'
        super().__init__(name=name)
        if entity_cell is None:
            if entity_embedding_dim is None:
                raise AttributeError('entity_embedding_dim should be given')
            if max_entity_num is None:
                raise AttributeError('max_entity_num should be given')
            entity_cell = EntityCell(max_entity_num=max_entity_num, entity_embedding_dim=entity_embedding_dim,
                                     name='entity_cell')
        self.entity_cell = entity_cell
        self.embedding_matrix = embedding_matrix
        self.sent_encoder_module = Sent_encoder()

    # @property
    # def variables(self):
    #     return self.trainable_variables+self.entity_cell.variables

    def call(self, inputs, keys, num_inputs=None, initial_entity_hidden_state=None,
             use_shared_keys=False, return_last=True, **kwargs):
        """
        Args:
            inputs: paragraph, paragraph mask in a list , paragraph of shape:[batch_size, max_sents_num, max_sents_len,
            keys: entity keys of shape : [batch_size, max_entity_num, entity_embedding_dim]
            num_inputs: ??? mask for keys??? is it needed in encoder?
            initial_entity_hidden_state
            use_shared_keys: bool
            return_last: if true, returns last state of entity hiddens, else returns all states
        """

        if len(inputs) != 2:
            raise AttributeError('expected 2 inputs but', len(inputs), 'were given')
        prgrph, prgrph_mask = inputs
        batch_size = prgrph.shape[0]
        max_sent_num = prgrph.shape[1]
        prgrph_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, prgrph)
        'prgrph_embeddings shape: [batch_size, max_sent_num, max_sent_len, embedding_dim]'
        encoded_sents = tf.zeros([batch_size, 1, prgrph_embeddings.shape[3]])
        for i in range(max_sent_num):
            ''' to see which sentences are available '''
            indices = tf.where(prgrph_mask[:, i, 0])
            # print('indices shape encode:, indices.shape)
            indices = tf.cast(tf.squeeze(indices, axis=1), tf.int32)
            current_sents = tf.gather(prgrph_embeddings[:, i, :, :], indices)
            # print('current_sents_call shape:', current_sents.shape)
            curr_encoded_sents = tf.expand_dims(self.sent_encoder_module(current_sents), axis=1)
            encoded_sents = tf.concat([encoded_sents, curr_encoded_sents], axis=1)

        encoded_sents = encoded_sents[:, 1:, :]
        sents_mask = prgrph_mask[:, :, 0]
        return self.entity_cell, simple_entity_network(entity_cell=self.entity_cell, inputs=[encoded_sents, sents_mask],
                                                       keys=keys,
                                                       initial_entity_hidden_state=initial_entity_hidden_state,
                                                       use_shared_keys=use_shared_keys,
                                                       return_last=return_last)


class RNNRecurrentEntityEncoder(tf.keras.Model):
    def __init__(self, num_units=None, entity_cell=None, rnn_cell=None, name=None, **kwargs):
        raise NotImplementedError

    def call(self, inputs, keys, num_inputs=None, num_keys=None,
             initial_hidden_state=None,
             initial_entity_hidden_state=None, update_positions=None, use_shared_keys=False,
             return_last=True, self_attention=False, **kwargs):
        raise NotImplementedError



