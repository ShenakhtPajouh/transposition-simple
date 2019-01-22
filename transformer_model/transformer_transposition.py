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


        self._separator = tf.get_variable('separator' , shape=[1,1,self._embedding_len] , dtype=tf.float64)

        with tf.variable_scope("transformer"):
            self._transformer = Transformer(MY_PARAMS)

        self.make_graph()

    def make_graph(self):
        self.bert_model = BertModel(config=self._config, is_training=False, input_ids=self.inputs,
                                    input_mask=self.mask,
                                    use_one_hot_embeddings=False, scope="bert")

        embeddings = self.bert_model.get_all_encoder_layers()[-1]

        bert_res = tf.scatter_nd(self.indices , embeddings , shape = tf.constant([self._batch_size , 2*self._max_sent_num+1 , self._embedding_len] , dtype = tf.int32))

        bert_res = tf.scatter_update(bert_res ,tf.squeeze(tf.reduce_sum(self.first_num_sents)) , tf.tile(self._separator , tf.constant([self._batch_size,1,1] , dtype=tf.int32)))

        transformer_padding = tf.dtypes.cast(tf.bitwise.invert(tf.sequence_mask(self.first_num_sents+self.second_num_sents , max_len = 2*self._max_sent_num , dtype=tf.int32)),tf.float64)
        transformer_res = self._transformer.encode(bert_res , transformer_padding)

        logits1 = tf.layers.dense(transformer_res , 32)
        final_logits = tf.layers.dense(logits1 , 2)

        return final_logits


















