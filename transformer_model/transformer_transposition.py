import tensorflow as tf
import json
from BERT.modeling import BertModel
from BERT.modeling import BertConfig
from transformer_model.model_params import MY_PARAMS
from transformer_model.transformer import Transformer



class trasformer(object):
    def __init__(self , name , embedding_len , bert_config_path , max_sent_num , batch_size):
        super(trasformer , self).__init__()

        self._name = name
        self._embedding_len = embedding_len
        self._max_sent_num = max_sent_num
        self._batch_size = batch_size

        with open(bert_config_path) as f:
            conf = json.loads(f.read())

        self._config = BertConfig(**conf)


        self.first_num_sents = tf.placeholder(dtype=tf.int32 , shape=[None] , name="first_num_sent")
        self.second_num_sents = tf.placeholder(dtype=tf.int32, shape=[None], name="second_num_sent")
        self.indices = tf.placeholder(dtype=tf.int32 , shape=[None] , name="first_indices")

        self.inputs = tf.placeholder(dtype=tf.int32 , shape=[None , None] ,name="inputs")
        self.mask = tf.placeholder(dtype=tf.int32 , shape=[None , None] , name="mask")

        self.target = tf.placeholder(dtype=tf.int32 , shape=[batch_size , 2] , name="targets")


        self._separator = tf.get_variable('separator' , shape=[1,1,self._embedding_len] , dtype=tf.float64)

        with tf.variable_scope("transformer"):
            self._transformer = Transformer(MY_PARAMS)

        self.make_graph()

    def make_graph(self):
        #encoding sentences with BERT
        self.bert_model = BertModel(config=self._config, is_training=False, input_ids=self.inputs,
                                    input_mask=self.mask,
                                    use_one_hot_embeddings=False, scope="bert")

        embeddings = self.bert_model.get_all_encoder_layers()[-1]

        #putting results of BERT into a tensor in a way that each pair of paragraphs would be in the same row
        bert_res = tf.scatter_nd(self.indices , embeddings , shape = tf.constant([self._batch_size , 2*self._max_sent_num+1 , self._embedding_len] , dtype = tf.int32))

        #adding separator between paragraphs
        tiled_separator = tf.tile(self._separator, tf.constant([self._batch_size, 1], dtype=tf.int32))

        separator_indices = tf.concat((tf.expand_dims(tf.range(self._batch_size), axis=1),
                                       tf.expand_dims(self.first_num_sents, axis=1)), axis=1)

        bert_res = tf.scatter_update(bert_res ,separator_indices , tiled_separator)

        #making padding for transformer
        transformer_padding = tf.dtypes.cast(tf.bitwise.invert(tf.sequence_mask(self.first_num_sents+self.second_num_sents+1 , max_len = 2*self._max_sent_num+1 , dtype=tf.int32)),tf.float64)

        #getting transformer result
        transformer_res = self._transformer.encode(bert_res , transformer_padding)

        #passing transformer result through a feedforward
        with tf.variable_scope("feedforward"):
            logits1 = tf.layers.dense(transformer_res , 32)
            self.final_logits = tf.layers.dense(logits1 , 2)
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.final_logits , labels= self.target)


    def variables(self):
        """
        :return: variables of the model, excluding the BERT model variables
        """
        ret = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transformer')
        ret = ret + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feedforward')

        return ret

















