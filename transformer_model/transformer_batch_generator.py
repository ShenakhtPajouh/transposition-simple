import numpy as np
from BERT.tokenization import FullTokenizer


def batch_generator(paragraphs, batch_size, max_sent_len, bert_vocab_path,
                    epochs):
    """
    yields a batch containing:
            first_num_sents: number of words in each first paragraph
            second_num_sents: number of words in each second paragraph
            indices: indices used for scatter_nd in the model
            inputs: np array in which each row contains ids of words of a sentence
            sentence_mask: mask for inputs
            labels: tells which paragraph is the first one

    :param paragraphs: list of triples (x,y,z) in which x , y are paragraphs and z is a label that indicates whether x is before y
    :param max_sent_len: maximum number of words that could be in a sentence
    """

    paragraphs = paragraphs[:(len(paragraphs) - (len(paragraphs) % batch_size))]
    tokenizer = FullTokenizer(bert_vocab_path)

    for epoch in range(epochs):
        for i in range(0, len(paragraphs), batch_size):
            print(i)
            paragraph_batch = paragraphs[i:i + batch_size]

            first_num_sents = np.zeros((batch_size), dtype=np.int32)
            second_num_sents = np.zeros((batch_size), dtype=np.int32)
            labels = np.zeros((batch_size, 2), dtype=np.int32)
            indices = []

            tokenized_sents = []
            num_sents = 0
            for j, (x, y, z) in enumerate(paragraph_batch):
                labels[j] = z
                x_ids = []
                for k, sent in enumerate(x):
                    tokenized = tokenizer.tokenize(sent)
                    x_ids.append(tokenizer.convert_tokens_to_ids(tokenized))
                    indices.append([j, k])

                y_ids = []
                for k, sent in enumerate(y):
                    tokenized = tokenizer.tokenize(sent)
                    y_ids.append(tokenizer.convert_tokens_to_ids(tokenized))
                    indices.append([j, k + len(x_ids) + 1])

                first_num_sents[j] = len(x_ids)
                second_num_sents[j] = len(y_ids)
                num_sents += len(x_ids) + len(y_ids)

                tokenized_sents = tokenized_sents + x_ids
                tokenized_sents = tokenized_sents + y_ids

            inputs = np.zeros((num_sents, max_sent_len), dtype=np.int32)
            sentence_mask = np.zeros((num_sents, max_sent_len), dtype=np.int32)

            for j, sent in enumerate(tokenized_sents):
                for k, word in enumerate(sent):
                    inputs[j, k] = word
                    sentence_mask[j, k] = 1

            yield first_num_sents, second_num_sents, np.array(
                indices, dtype=np.int32), inputs, sentence_mask, labels
