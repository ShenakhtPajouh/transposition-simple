import numpy as np
import tensorflow as tf
import json
from BERT import tokenization
from BERT import modeling
import utils


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
        """
        :param batch: a list of paragraphs to be encoded
        :return: - a numpy array of shape [batch_size*maximum_num_tokens]
                 - a list of lists of tokens for each paragraph
        """

        tokenized = []
        tokenized_ids = []
        for seq in batch:
            tokens = self._tokenizer.tokenize(seq)
            ids = self._tokenizer.convert_tokens_to_ids(tokens)

            tokenized.append(tokens)
            tokenized_ids.append(ids)

        max_len = -1
        for ls in tokenized_ids:
            max_len = max (max_len , len(ls))

        ins = np.zeros((len(batch),max_len) , dtype=np.int32)
        mask = np.zeros((len(batch),max_len) , dtype=np.int32)

        for i,ids in enumerate(tokenized_ids):
            for j, id in enumerate(ids):
                ins[i][j]=ids[j]
                mask[i][j]=1

        embds = self._sess.run(self._embeddings, {self._inputs: ins, self._mask: mask })

        final_embd = embds[-1]+embds[-2]+embds[-3]+embds[-4]

        return tokenized , final_embd



def get_normal_batch(path , bert_config_path , bert_vocab_path , bert_base_path , batch_size):
    paragraphs = utils.get_from_file(path)
    encoder = BERT_encoder(bert_config_path , bert_vocab_path , bert_base_path)












def get_batch(path , batch_size , mode):
    if mode=='normal':
        get_normal_batch(path , batch_size)