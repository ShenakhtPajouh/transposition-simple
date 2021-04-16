import tensorflow as tf
import numpy as np
import json
import utils
from BERT import tokenization
from BERT import modeling
import time


class BERT_encoder:
    """
    a class for encoding paragraphs
    """

    def __init__(self, config_path, vocab_path, base_path):
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
        self._model = modeling.BertModel(config=config,
                                         is_training=False,
                                         input_ids=self._inputs,
                                         input_mask=self._mask,
                                         use_one_hot_embeddings=False,
                                         scope="bert")

        self._embeddings = self._model.get_all_encoder_layers()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)

        saver = tf.train.Saver()
        saver.restore(self._sess, self._base_path)

    def encode(self, batch):
        """
        :param batch: a list of sequences to be encoded
        :return: - a numpy array of shape [batch_size,embedding(768)]
        """

        tokenized_ids = []
        for x, y in batch:
            ids = [101] + x + [102] + y

            tokenized_ids.append(ids)

        max_len = -1
        for ls in tokenized_ids:
            max_len = max(max_len, len(ls))

        ins = np.zeros((len(batch), max_len), dtype=np.int32)
        mask = np.zeros((len(batch), max_len), dtype=np.int32)

        for i, ids in enumerate(tokenized_ids):
            for j, id in enumerate(ids):
                ins[i][j] = ids[j]
                mask[i][j] = 1

        embds = self._sess.run(self._embeddings, {
            self._inputs: ins,
            self._mask: mask
        })

        final_embd = embds[-1][:, 0, :] + embds[-2][:, 0, :] + embds[
            -3][:, 0, :] + embds[-4][:, 0, :]

        return final_embd


def make_dataset(paragraphs, bert_config_path, bert_vocab_path, bert_base_path,
                 batch_size, max_len):
    """
        a makes two np arrays data and labels
        data: [,768] np array containing [cls] token encoding used for classification
        labels: [,2] np array that determines whether the first paragraph has occured first
    """

    data = np.zeros((2 * len(paragraphs), 768), dtype=np.float64)
    labels = np.zeros((2 * len(paragraphs), 2), dtype=np.float64)

    encoder = BERT_encoder(bert_config_path, bert_vocab_path, bert_base_path)

    count = 0
    batch = []
    start = time.time()
    mem = 0
    for x, y in paragraphs:
        batch.append(
            (x[max(0,
                   len(x) - max_len // 2):], y[0:min(len(y), max_len // 2)]))
        count += 1
        if (count % batch_size == 0):
            print(count)
            mem = count
            embds = encoder.encode(batch)
            data[count - batch_size:count] = embds
            labels[count - batch_size:count, 0] = 1
            batch = []
            end = time.time()
            print(end - start)
            start = time.time()

    count = mem
    batch = []
    for x, y in paragraphs:
        batch.append(
            (y[max(0,
                   len(y) - max_len // 2):], x[0:min(len(x), max_len // 2)]))
        count += 1
        if (count % batch_size == 0):
            print(count)
            embds = encoder.encode(batch)
            data[count - batch_size:count] = embds
            labels[count - batch_size:count, 1] = 1
            batch = []
            end = time.time()
            print(end - start)
            start = time.time()

    print(data[:80])
    data = data[:count]
    labels = labels[:count]

    permutation = np.random.permutation(data.shape[0])

    data = data[permutation]
    labels = labels[permutation]

    return data, labels
