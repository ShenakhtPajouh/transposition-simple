import numpy as np
import tensorflow as tf
import json
from BERT import tokenization
from BERT import modeling


class BERT_encoder:
    """
    a class for encoding paragraphs
    """
    def __init__(self , config_path , vocab_path , base_path):
        """
        making instance of encoder and bert model
        """

        self._base_path = base_path

        self._tokenizer = tokenization.FullTokenizer(vocab_path)

        with open(config_path) as f:
            self._conf = json.loads(f.read())

        config = modeling.BertConfig(**self._conf)

        self._inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self._mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self._model = modeling.BertModel(config=config, is_training=False, input_ids=self._inputs, input_mask=self._mask,
                                   use_one_hot_embeddings=False)

        self._embeddings = self._model.get_all_encoder_layers()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)

        saver = tf.train.Saver()
        saver.restore(self._sess, self._base_path)

    def encode(self ,batch):
        #باگ داره!!!!!
        """
        :param batch: a list of paragraphs to be encoded
        :return: a numpy array of shape []
        """
        tokenized = []
        for seq in batch:
            tokens = self._tokenizer.tokenize(seq)
            ids = self._tokenizer.convert_tokens_to_ids(tokens)

            tokenized.append(ids)

        max_len = -1
        for ls in tokenized:
            max_len = max (max_len , len(ls))

        ins = np.zeros((len(batch),max_len) , dtype=np.int32)
        mask = np.zeros((len(batch),max_len) , dtype=np.int32)

        for i,ids in enumerate(tokenized):
            for j, id in enumerate(ids):
                ins[i][j]=ids[j]
                mask[i][j]=1

        embds = self._sess.run(self._embeddings, {self._inputs: ins, self._mask: mask }).eval()




def make_bert_encoder(config_path , vocab_path , base_path):
    tokenizer = tokenization.FullTokenizer(vocab_path)

    with open("conf/bert_config.json") as f:
        conf = json.loads(f.read())

    config = modeling.BertConfig(**conf)

    inputs = tf.placeholder(shape=[None, None], dtype=tf.int32)
    mask = tf.placeholder(shape=[None, None], dtype=tf.int32)
    model = modeling.BertModel(config=config, is_training=False, input_ids=inputs, input_mask=mask,
                               use_one_hot_embeddings=False)

    embeddings = model.get_all_encoder_layers()



def encode(seq , tokenizer , model):



def get_concat_batch(path , batch_size):
    paragraphs = utils.get_from_file(path)

    for x , y in paragraphs:







def get_batch(path , batch_size , mode):
    if mode=='concat':
        get_concat_batch(path , batch_size)