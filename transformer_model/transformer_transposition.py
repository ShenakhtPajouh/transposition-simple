import tensorflow as tf
import json
from BERT.modeling import BertModel
from BERT.modeling import BertConfig
from transformer_model.model_params import MY_PARAMS
from transformer_model.transformer import Transformer

_NEG_INF = -1e9

class transformer(object):
    def __init__(self , name , embedding_len , bert_config_path , max_sent_num , batch_size , train):
        super(transformer , self).__init__()

        self._name = name
        self._embedding_len = embedding_len
        self._max_sent_num = max_sent_num
        self._batch_size = batch_size
        self._train = train

        with open(bert_config_path) as f:
            conf = json.loads(f.read())

        self._config = BertConfig(**conf)

        with tf.device('/gpu:0'):
            self.first_num_sents = tf.placeholder(dtype=tf.int32 , shape=[batch_size] , name="first_num_sent")
            self.second_num_sents = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="second_num_sent")
            self.indices = tf.placeholder(dtype=tf.int32 , shape=[None,None] , name="first_indices")

            self.inputs = tf.placeholder(dtype=tf.int32 , shape=[None , None] ,name="inputs")
            self.mask = tf.placeholder(dtype=tf.int32 , shape=[None , None] , name="mask")

            self.target = tf.placeholder(dtype=tf.int32 , shape=[batch_size , 2] , name="targets")


            with tf.variable_scope('separator'):
                self._separator = tf.get_variable('separator' , shape=[1,self._embedding_len] , dtype=tf.float32)


            self._transformer = Transformer(MY_PARAMS , train)

        self.make_graph()

    def make_graph(self):
        #encoding sentences with BERT
        self.bert_model = BertModel(config=self._config, is_training=False, input_ids=self.inputs,
                                    input_mask=self.mask,
                                    use_one_hot_embeddings=False, scope="bert")
        with tf.device('/gpu:0'):
            embeddings = tf.reduce_mean(self.bert_model.get_all_encoder_layers()[-1],axis=1)

            tiled_separator = tf.tile(self._separator, tf.constant([self._batch_size, 1], dtype=tf.int32))
            separator_indices = tf.concat((tf.expand_dims(tf.range(self._batch_size), axis=1),
                                           tf.expand_dims(self.first_num_sents, axis=1)), axis=1)
            all_embeddings = tf.concat((embeddings , tiled_separator) , axis = 0)
            all_indices = tf.concat((self.indices , separator_indices) , axis = 0)

            #putting results of BERT into a tensor in a way that each pair of paragraphs would be in the same row
            bert_res = tf.scatter_nd(all_indices , all_embeddings , shape = tf.constant([self._batch_size , 2*self._max_sent_num+1 , self._embedding_len] , dtype = tf.int32))

            #making padding for transformer
            transformer_padding = tf.dtypes.cast(tf.bitwise.invert(tf.sequence_mask(self.first_num_sents+self.second_num_sents+1 ,maxlen = 2*self._max_sent_num+1 , dtype=tf.int32)),tf.float32)

            attention_bias = tf.expand_dims(
            tf.expand_dims(transformer_padding*_NEG_INF, axis=1), axis=1)

            #getting transformer result

        with tf.variable_scope("transformer"):
            transformer_res = self._transformer.encode((bert_res,transformer_padding) , attention_bias)

        transformer_encoded = tf.reduce_mean(transformer_res , axis=1)

        with tf.device('/gpu:0'):
            #passing transformer result through a feedforward
            with tf.variable_scope("feedforward"):
                logits1 = tf.layers.dense(transformer_encoded , 32)
                self.final_logits = tf.layers.dense(logits1 , 2)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.final_logits , labels= self.target))


    def variables(self):
        """
        :return: variables of the model, excluding the BERT model variables
        """
        ret = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transformer')
        ret = ret + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feedforward')
        ret = ret + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='separator')

        return ret

















