import numpy as np
import pickle
from remote.API import get_paragraphs
import numpy as np


def extract_embedding_from_dict(dict_path, vocab_size, embedding_len, vectors_path, stoi_path, itos_path):
    """
    :param dict_path: path of the dictionary of word to embedding
    :param vocab_size: size of embedding vocabulary
    :param embedding_len: length of embedding vector
    """

    # an np array of shape [vocab_size , embedding_len] containing embedding vectors
    vectors = np.zeros((vocab_size, embedding_len), dtype=np.float64)
    # a dictionary that maps words to their corresponding index in vectors
    stoi = dict()
    # a list in which index of each word is its index in vectors
    itos = []

    i = 0
    with open(dict_path, 'r', encoding='UTF-8') as infile:
        for line in infile.readlines():
            row = line.strip().split(' ')
            itos.append(row[0])
            listt = [float(j) for j in row[1:]]
            stoi[row[0]] = i
            vectors[i] = np.array(listt)
            i += 1

    with open(vectors_path, 'wb') as pkl:
        pickle.dump(vectors, pkl)
    with open(stoi_path, 'wb') as pkl:
        pickle.dump(stoi, pkl)
    with open(itos_path, 'wb') as pkl:
        pickle.dump(itos, pkl)


def get_from_file(path):
    """
    :param path: path of the file
    :return: object read from file
    """
    with open(path, 'rb') as pkl:
        ret = pickle.load(pkl, encoding='iso-8859-1')

    return ret


def get_paragraphs_as_words(stoi, paragraph_id=None, books=None, tags=None,
                            num_sequential=2, Paragraph_Object=True):
    """
    :return: a triple of np arrays: first: embedding of words of first paragraphs per row
                                    second: embedding of words of second paragraphs per row
                                    labels: whether the first paragraph has happend before the second paragraph
    """

    paragraphs = get_paragraphs(paragraph_id=paragraph_id, books=books,
                                tags=tags, num_sequential=num_sequential, paragraph_object=Paragraph_Object)

    first_seq = []
    second_seq = []

    max_len = -1
    max_ind = 0

    for x, y in paragraphs:
        first_seq.append(x.text(format="words", lowercase=True))
        second_seq.append(y.text(format="words", lowercase=True))

    for i, seq in enumerate(first_seq):
        if max_len < len(seq):
            max_len = len(seq)
            max_ind = i

    for i, seq in enumerate(second_seq):
        if max_len < len(seq):
            max_len = len(seq)
            max_ind = i

    print(" ".join(first_seq[max_ind]))
    print(" ")
    print(" ".join(second_seq[max_ind]))

    first = np.zeros((2 * len(paragraphs), max_len), dtype=np.int32)
    second = np.zeros((2 * len(paragraphs), max_len), dtype=np.int32)

    labels = np.zeros((2 * len(paragraphs), 2), dtype=np.float32)
    labels[:len(paragraphs), 0] = 1
    labels[len(paragraphs):, 1] = 1
    count = 0

    for i, seq in enumerate(first_seq):
        for j, word in enumerate(seq):
            if word in stoi:
                first[i, j] = stoi[word]
            else:
                count += 1
                first[i, j] = stoi['**BLANK**']

    for i, seq in enumerate(second_seq):
        for j, word in enumerate(seq):
            if word in stoi:
                second[i, j] = stoi[word]
            else:
                count += 1
                second[i, j] = stoi['**BLANK**']

    print(max_len)

    first[len(paragraphs):, :] = second[:len(paragraphs), :]
    second[len(paragraphs):, :] = first[:len(paragraphs), :]

    permutation = np.random.permutation(2 * len(paragraphs))

    first = first[permutation]
    second = second[permutation]
    labels = labels[permutation]

    with open('first.pkl', 'wb') as pkl:
        pickle.dump(first, pkl)
    with open('second.pkl', 'wb') as pkl:
        pickle.dump(second, pkl)
    with open('labels.pkl', 'wb') as pkl:
        pickle.dump(labels, pkl)

    return first, second, labels









