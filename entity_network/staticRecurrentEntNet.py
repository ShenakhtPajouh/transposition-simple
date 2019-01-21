import tensorflow as tf
from tensorflow.contrib import autograph

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
        # I assume word embedding for indexes greater that sentnece length is zero vector, so it does not effect sentence encoding '
        assert isinstance(inputs, list)
        sents = inputs[0]

        return tf.reduce_sum(sents, 1)


class Update_entity(tf.keras.Model):
    def __init__(self, entity_num, entity_embedding_dim, activation=tf.nn.relu, initializer=None, name=None):
        if name is None:
            name = 'update_entity'

        super().__init__(name=name)
        self.entity_num = entity_num
        self.entity_embedding_dim = entity_embedding_dim
        self.activation = activation
        if initializer is None:
            self.initializer = tf.keras.initializers.random_normal()
        else:
            self.initializer = initializer
        # defining Variables
        self.U = None
        # self._variables.append(self.U)
        self.V = None
        # self._variables.append(self.V)
        self.W = None
        # self._variables.append(self.W)
        self.built = False

    def build(self, input_shape):
        shape = [self.entity_embedding_dim, self.entity_embedding_dim]
        self.U = K.variable(self.initializer(shape), name='U')
        self.V = K.variable(self.initializer(shape), name='V')
        self.W = K.variable(self.initializer(shape), name='W')
        self.built = True

    def initialize_hidden(self, hiddens):
        self.batch_size = hiddens.shape[0]
        self.hiddens = hiddens

    def assign_keys(self, entity_keys):
        self.keys = entity_keys

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
        # expanded=tf.expand_dims(encoded_sents,axis=1)
        # print('expanded shape:', expanded.shape)
        # print('tile shape:', tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]).shape)
        # print('curent hiddens shape:', current_hiddens.shape)
        #
        # print(tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_hiddens)\
        #        +tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_keys),axis=2).shape)
        # return tf.sigmoid(tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_hiddens)\
        #        +tf.multiply(tf.tile(tf.expand_dims(encoded_sents,axis=1),[1,self.entity_num,1]),current_keys),axis=2))

        # break complex formulas to simpler to be trackable!!
        print('enocded_sents dtype:',encoded_sents.dtype)
        print('current_hiddens dtype:',current_hiddens.dtype)
        return tf.sigmoid(tf.reduce_sum(tf.multiply(tf.expand_dims(encoded_sents, 1), current_hiddens) +
                                        tf.multiply(tf.expand_dims(encoded_sents, 1), current_keys), axis=2))

    def update_hidden(self, gates, current_hiddens, current_keys, encoded_sents, indices):
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
                       tf.matmul(tf.reshape(tf.tile(tf.expand_dims(encoded_sents, 1), [1, self.entity_num, 1]),
                                            shape=[-1, self.entity_embedding_dim]), self.W),
                       shape=[curr_prgrphs_num, self.entity_num, self.entity_embedding_dim]))
        'h_tilda shape: [current_prgrphs_num, entity_num, entity_embedding_dim]'
        # tf.multiply(gates,h_tilda)
        self.hiddens = self.hiddens + tf.scatter_nd(tf.expand_dims(indices, axis=1), tf.multiply(
            tf.tile(tf.expand_dims(gates, axis=2), [1, 1, self.entity_embedding_dim]), h_tilda),
                                                    shape=[self.batch_size, self.entity_num, self.entity_embedding_dim])

    def normalize(self):
        self.hiddens = tf.nn.l2_normalize(self.hiddens, axis=2)

    def call(self, inputs, training=None):
        """
        Description:
            Updates related etities
        Args:
            inputs: encoded_sents shape : [current_prgrphs_num,encoding_dim] , here encoding_dim is equal to embedding_dim
        """
        assert isinstance(inputs, list)

        encoded_sents, indices = inputs
        current_hiddens = tf.gather(self.hiddens, indices)
        print('ENCODE')
        print(self.keys.shape)
        print(indices)
        current_keys = tf.gather(self.keys, indices)

        if current_hiddens.shape != current_keys.shape:
            raise AttributeError('hiddens and kes must have same shape')

        gates = self.get_gate(encoded_sents, current_hiddens, current_keys)
        self.update_hidden(gates, current_hiddens, current_keys, encoded_sents, indices)
        self.normalize()
        return self.hiddens


class StaticRecurrentEntNet(tf.keras.Model):
    def __init__(self, entity_num, entity_embedding_dim, rnn_hidden_size,max_sent_num, name=None):

        if name is None:
            name = 'staticRecurrentEntNet'
        super().__init__(name=name)
        # embedding_matrix shape: [vocab_size, embedding_dim]
        # I assume the last row is an all zero vector for fake words with index embedding_matrix.shape[0]
        # self.add_zero_vector()
        self.entity_num = entity_num
        self.entity_embedding_dim = entity_embedding_dim
        self.hidden_Size = rnn_hidden_size
        self.max_sent_num = max_sent_num
        'start_token shape:[1,enbedding_dim]'

        ' defining submodules '
        self.sent_encoder_module = Sent_encoder()
        self.update_entity_module = Update_entity(self.entity_num, self.entity_embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.hidden_Size, return_state=True)
        self.entity_dense = tf.keras.layers.Dense(self.hidden_Size)
        self.start_hidden_dense=tf.keras.layers.Dense(self.hidden_Size)


    def build(self, input_shape):
        self.entity_attn_matrix = K.random_normal_variable(shape=[self.hidden_Size, self.embedding_dim],
                                                           mean=0, scale=0.05, name='entity_attn_matrix')

    # @property
    # def trainable(self):
    #     return self._trainable

    def attention_hiddens(self, query, keys, memory_mask):
        '''
        Description:
            attention on keys with given quey, value is equal to keys
        Args:
            inputs: query shape: [curr_prgrphs_num, hiddens_size]
                    keys shape: [curr_prgrphs_num, prev_hiddens_num, hidden_size]
                    memory_mask: [curr_prgrphs_num, prev_hiddens_num]
            output shape: [curr_prgrphs_num, hidden_size]
        '''
        print('in attention_hiddens')
        print('keys shape:',keys.shape)
        print('query shape:',query.shape)
        print('mask shape',memory_mask.shape)
        values = tf.identity(keys)
        query_shape = tf.shape(query)
        keys_shape = tf.shape(keys)
        values_shape = tf.shape(values)
        batch_size = query_shape[0]
        seq_length = keys_shape[1]
        query_dim = query_shape[1]
        indices = tf.where(memory_mask)
        queries = tf.gather(query, indices[:, 0])
        keys = tf.boolean_mask(keys, memory_mask)
        attention_logits = tf.reduce_sum(tf.multiply(queries, keys), axis=-1)
        # print('attention logits:',attention_logits)
        # print('tf.where(memory_mask):',tf.where(memory_mask))
        attention_logits = tf.scatter_nd(tf.where(memory_mask), attention_logits, [batch_size, seq_length])
        attention_logits = tf.where(memory_mask, attention_logits, tf.fill([batch_size, seq_length], -float("Inf")))
        attention_coefficients = tf.nn.softmax(attention_logits, axis=1)
        attention = tf.expand_dims(attention_coefficients, -1) * values

        # print(tf.reduce_sum(attention,1))

        return tf.reduce_sum(attention, 1)

    def attention_entities(self, query, entities):
        '''
        Description:
            attention on entities
        Arges:
            inputs: query shape: [curr_prgrphs_num, hiddens_size]
                    entities shape: [curr_prgrphs_num, entities_num, entitiy_embedding_dim]
            output shape: [curr_prgrphs_num, entity_embedding_dim]
        '''

        return tf.reduce_sum(tf.multiply(tf.expand_dims(tf.matmul(query, self.entity_attn_matrix), axis=1), entities),
                             axis=1)

    def calculate_hidden(self, curr_sents_prev_hiddens, entities, mask):
        """
        Description:
            calculates current hidden state that should be fed to lstm for predicting the next word, with attention on previous hidden states, THEN entities
        Args:
            inputs: curr_sents_prev_hiddens shape: [curr_prgrphs_num, prev_hiddens_num, hidden_size]
                    entities: [curr_prgrphs_num, entities_num, entity_embedding_dim]
                    mask: [curr_prgrphs_num, prev_hiddens_num]
            output shape: [curr_prgrphs_num, hidden_size]
        """

        """
        attention on hidden states:
            query: last column (last hidden_state)
            key and value: prev_columns
        """
        attn_hiddens_output = self.attention_hiddens(
            curr_sents_prev_hiddens[:, curr_sents_prev_hiddens.shape[1] - 1, :],
            curr_sents_prev_hiddens[:, :curr_sents_prev_hiddens.shape[1], :], mask)
        attn_entities_output = self.attention_entities(attn_hiddens_output, entities)
        return self.entity_dense(attn_entities_output)

    # def add_zero_vector(self):
    #     embedding_dim=self.embedding_matrix.shape[1]
    #     self.embedding_matrix=tf.concat([self.embedding_matrix,tf.zeros([1,embedding_dim])],axis=0)

    @autograph.convert()
    def encode_prgrph(self, inputs):
        '''
            TASK 1
            ENCODING given paragraph
        '''

        ''' 
        inputs: entity_keys, prgrph, prgrph_mask
        output: entity_hiddens last state
        '''

        # if prgrph is None:
        #     raise AttributeError('prgrph is None')
        # if prgrph_mask is None:
        #     raise AttributeError('prgrph_mask is None')
        # if entity_keys is None:
        #     raise AttributeError('entity_keys is None')
        if len(inputs) != 3:
            raise AttributeError('expected 3 inputs but', len(inputs), 'were given')
        entity_keys, prgrph_embeddings, prgrph_mask = inputs
        batch_size = prgrph_embeddings.shape[0]
        max_sent_num = prgrph_embeddings.shape[1]

        # print('first_prgrph_embedding shape:',first_prgrph_embeddings.shape)
        'prgrph_embeddings shape: [batch_size, max_sent_num, max_sent_len, embedding_dim]'

        self.update_entity_module.initialize_hidden(
            tf.zeros([batch_size, self.entity_num, self.entity_embedding_dim], tf.float32))
        self.update_entity_module.assign_keys(entity_keys)

        for i in range(max_sent_num):
            ''' to see which sentences are available '''
            indices = tf.where(prgrph_mask[:, i])
            print('indices shape encode:',indices.shape)
            indices = tf.cast(tf.squeeze(indices, axis=1),tf.int32)
            print('indices_p1_mask', indices)
            # print('first_prgrph_embedding shape:',first_prgrph_embeddings.shape)
            # print('first_prgrph_embeddings[:,i,:,:] shape:',first_prgrph_embeddings[:,i,:,:].shape)
            encoded_sents = tf.gather(prgrph_embeddings[:, i, :], indices)
            # print('current_sents_call shape:', current_sents.shape)
            # encoded_sents = self.sent_encoder_module([current_sents])
            self.update_entity_module([encoded_sents, indices])

        return self.update_entity_module.hiddens


    def call(self, inputs, training=None):
        '''
        args:
            inputs: mode: encode, decode_train, decode_test
                    prgrph shape : [batch_size, max_sent_num, max_sent_len]
                    * I assume that fake words have index equal to embedding_matrix.shape[0]
                    entity_keys : initialized entity keys of shape : [batch_size, entity_num, entity_embedding_dim] , entity_embedding_dim=embedding_dim for now
                    prgrph_mask : mask for given prgrph, shape=[batch_size, max_sent_num, max_sent_len]
        '''

        print('inputs type:', type(inputs))
        assert isinstance(inputs, list)
        # what is inputs?
        mode = inputs[0]
        return self.encode_prgrph(inputs[1:])

