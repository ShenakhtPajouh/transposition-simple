import tensorflow as tf
import numpy as np
import json
import tokenization
import modeling
import time

encoder = None

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
        self._model = modeling.BertModel(config=config, is_training=False, input_ids=self._inputs,
                                         input_mask=self._mask,
                                         use_one_hot_embeddings=False, scope="bert")

        self._embeddings = self._model.get_all_encoder_layers()
        print("\n".join([v.name for v in tf.global_variables()]))
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
        for seq in batch:
            tokens = self._tokenizer.tokenize(seq)
            ids = self._tokenizer.convert_tokens_to_ids(tokens)

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

        embds = self._sess.run(self._embeddings, {self._inputs: ins, self._mask: mask})

        final_embd = embds[-1]

        return np.mean(final_embd, axis=1)


def make_normal_dataset(paragraphs, bert_config_path, bert_vocab_path, bert_base_path, batch_size):
    max_sent = -1
    for x, y in paragraphs:
        max_sent = max(max_sent, len(x), len(y))

    offset = (len(paragraphs) - len(paragraphs) % batch_size)

    first = np.zeros((2 * len(paragraphs), max_sent, 768), dtype=np.float64)
    second = np.zeros((2 * len(paragraphs), max_sent, 768), dtype=np.float64)
    labels = np.zeros((2 * len(paragraphs), 2), dtype=np.float64)
    first_sent_num = np.zeros((2 * len(paragraphs)), dtype=np.int32)
    second_sent_num = np.zeros((2 * len(paragraphs)), dtype=np.int32)

    encoder = BERT_encoder(bert_config_path, bert_vocab_path, bert_base_path)

    start = time.time()

    for i in range(0, offset+1, batch_size):
        print (i)
        if (i<len(paragraphs)):
            paragraph_batch = paragraphs[i:min(i + batch_size , len(paragraphs))]

            sent_batch = []
            for x, y in paragraph_batch:
                sent_batch = sent_batch + x + y

            embds = encoder.encode(sent_batch)

            count = 0
            for j, (x, y) in enumerate(paragraphs[i:i + batch_size]):
                first_sent_num[i + j] = len(x)
                second_sent_num[i + j] = len(y)
                first_sent_num[len(paragraphs) + i + j] = len(y)
                second_sent_num[len(paragraphs) + i + j] = len(x)

                for k in range(len(x)):
                    first[i + j, k] = embds[count]
                    second[len(paragraphs) + i + j, k] = embds[count]
                    count += 1
                for k in range(len(y)):
                    second[i + j, k] = embds[count]
                    first[len(paragraphs) + i + j, k] = embds[count]
                    count += 1
            end = time.time()
            print(end - start)
            start = time.time()

        labels[:len(paragraphs), 0] = 1
        labels[len(paragraphs):, 1] = 1

    return first, second, first_sent_num, second_sent_num, labels




def make_dataset(paragraphs, bert_config_path, bert_vocab_path, bert_base_path, batch_size):
    """
    ::param:: path: path two a file containing tuples of paragraphs as lists of lists of words

    desc:
        makes two np arrays: data and labels
        data: [,768] np array containing [cls] token encoding used for classification
        labels: [,2] np array that determines whether the first paragraph has occured first
        """

    return make_normal_dataset(paragraphs, bert_config_path, bert_vocab_path, bert_base_path, batch_size)
