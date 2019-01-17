import numpy as np
import pickle
from random import shuffle
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


def get_paragraphs_as_words(stoi, max_len=512 , min_len=10 , min_sent=3, paragraph_id=None, books=None, tags=None,
                            num_sequential=2, Paragraph_Object=True):
    """
    :return: a triple of np arrays: first: embedding of words of first paragraphs per row
                                    second: embedding of words of second paragraphs per row
                                    labels: whether the first paragraph has happend before the second paragraph
    """

    paragraphs = get_paragraphs(paragraph_id=paragraph_id, books=books,
                                tags=tags, num_sequential=num_sequential, paragraph_object=Paragraph_Object)

    paragraphs = filter(paragraphs, max_len, min_len, min_sent)

    first_seq = []
    second_seq = []

    max_len = -1

    for x, y in paragraphs:
        first_seq.append(x.text(format="words", lowercase=True))
        second_seq.append(y.text(format="words", lowercase=True))

    for seq in first_seq:
        max_len = max (max_len , len(seq))

    for seq in second_seq:
        max_len = max (max_len , len(seq))

    first = np.zeros((2 * len(paragraphs), max_len), dtype=np.int32)
    second = np.zeros((2 * len(paragraphs), max_len), dtype=np.int32)

    labels = np.zeros((2 * len(paragraphs), 2), dtype=np.float32)
    labels[:len(paragraphs), 0] = 1
    labels[len(paragraphs):, 1] = 1

    for i, seq in enumerate(first_seq):
        for j, word in enumerate(seq):
            if word in stoi:
                first[i, j] = stoi[word]
            else:
                first[i, j] = stoi['**BLANK**']

    for i, seq in enumerate(second_seq):
        for j, word in enumerate(seq):
            if word in stoi:
                second[i, j] = stoi[word]
            else:
                second[i, j] = stoi['**BLANK**']


    first[len(paragraphs):, :] = second[:len(paragraphs), :]
    second[len(paragraphs):, :] = first[:len(paragraphs), :]

    permutation = np.random.permutation(2 * len(paragraphs))

    first = first[permutation]
    second = second[permutation]
    labels = labels[permutation]

    return first, second, labels


def make_dataset_with_indices(stoi, max_len=512 , min_len=10 , min_sent=3, paragraph_id=None, books=None, tags=None,
                              num_sequential=2, Paragraph_Object=True):
    """
        :return: a tuple of four np arrays: first: embedding of words of first paragraphs per row
                                        second: embedding of words of second paragraphs per row
                                        labels: whether the first paragraph has happend before the second paragraph
                                        indices: indices in which each sentence of paragraph ends
        """

    paragraphs = get_paragraphs(paragraph_id=paragraph_id, books=books,
                                tags=tags, num_sequential=num_sequential, paragraph_object=Paragraph_Object)

    paragraphs = filter(paragraphs, max_len, min_len, min_sent)

    first_sents = []
    second_sents = []

    max_len = -1
    max_sent = -1

    for x, y in paragraphs:
        first_sents.append(x.text(format="sentences", lowercase=True))
        second_sents.append(y.text(format="sentences", lowercase=True))

    first_seq = []
    second_seq = []

    for i, seq in enumerate(first_sents):
        max_sent = max(max_sent, len(seq))
        l = []
        for sent in seq:
            l += sent
        max_len = max(max_len, len(l))
        first_seq.append(l)

    for i, seq in enumerate(second_sents):
        max_sent = max(max_sent, len(seq))
        l = []
        for sent in seq:
            l += sent
        max_len = max(max_len, len(l))
        second_seq.append(l)

    first = np.zeros((2 * len(paragraphs), max_len), dtype=np.int64)
    second = np.zeros((2 * len(paragraphs), max_len), dtype=np.int64)

    labels = np.zeros((2 * len(paragraphs), 2), dtype=np.float64)
    labels[:len(paragraphs), 0] = 1
    labels[len(paragraphs):, 1] = 1

    indices1 = np.ones((2 * len(paragraphs), max_sent), dtype=np.int64) * (max_len + 1)
    indices2 = np.ones((2 * len(paragraphs), max_sent), dtype=np.int64) * (max_len + 1)


    for i, p in enumerate(first_sents):
        l = -1
        for j, sent in enumerate(p):
            l += len(sent)
            indices1[i][j] = l
            indices2[len(paragraphs) + i, j] = l

    for i, p in enumerate(second_sents):
        l = -1
        for j, sent in enumerate(p):
            l += len(sent)
            indices2[i][j] = l
            indices1[len(paragraphs) + i, j] = l

    for i, seq in enumerate(first_seq):
        for j, word in enumerate(seq):
            if word in stoi:
                first[i, j] = stoi[word]
            else:
                first[i, j] = stoi['**BLANK**']

    for i, seq in enumerate(second_seq):
        for j, word in enumerate(seq):
            if word in stoi:
                second[i, j] = stoi[word]
            else:
                second[i, j] = stoi['**BLANK**']

    first[len(paragraphs):, :] = second[:len(paragraphs), :]
    second[len(paragraphs):, :] = first[:len(paragraphs), :]

    permutation = np.random.permutation(2 * len(paragraphs))

    first = first[permutation]
    second = second[permutation]
    labels = labels[permutation]
    indices1 = indices1[permutation]
    indices2 = indices2 [permutation]


    return first, second, labels, indices1, indices2


def write_paragraphs_in_file(file_path , splits , max_len=512 , min_len=10 , min_sent=3 , paragraph_id=None, books=None, tags=None,
                            num_sequential=2, Paragraph_Object=True ):
    paragraphs = get_paragraphs(paragraph_id=paragraph_id, books=books,
                                tags=tags, num_sequential=num_sequential, paragraph_object=Paragraph_Object)

    paragraphs = filter(paragraphs , max_len , min_len , min_sent)

    prev=0
    paragraphs_split=[]

    for i in splits:
        print (i)
        paragraphs_split.append(paragraphs[prev:min(len(paragraphs),int(prev+len(paragraphs)*i))])
        prev = int(prev+len(paragraphs)*i)

    for i,split in enumerate(paragraphs_split):
        with open(file_path+'-'+str(i)+'.txt', 'a') as outfile:
             x = x.text(format="text")
             y = y.text(format="text")
             outfile.write(x+'\n')
             outfile.write(y+'\n')




def filter (paragraphs , max_len=512 , min_len=10 , min_sent=3):
    ret = []
    for tuple in paragraphs:
        flag=False
        for seq in tuple:
            l = len(seq.text(format="words"))
            ls = len(seq.text(format="sentences"))
            if (l>max_len or l<min_len or ls<min_sent):
                flag = True

        if (flag==False):
            ret.append(tuple)

    return ret




def get_paragraphs_as_sents(stoi, max_len=512 , min_len=10 , min_sent=3 ,paragraph_id=None, books=None, tags=None,
                            num_sequential=2, Paragraph_Object=True):

    paragraphs = get_paragraphs(paragraph_id=paragraph_id, books=books,
                                tags=tags, num_sequential=num_sequential, paragraph_object=Paragraph_Object)

    paragraphs=filter(paragraphs , max_len , min_len , min_sent)

    print (len(paragraphs))

    first_seq = []
    second_seq = []

    max_p_len = -1
    max_sent_len = -1

    for x, y in paragraphs:
        first_seq.append(x.text(format="sentences", lowercase=True))
        second_seq.append(y.text(format="sentences", lowercase=True))

    for paragraph in first_seq:
        max_p_len = max(max_p_len , len(paragraph))
        for sent in paragraph:
            max_sent_len = max(max_sent_len , len(sent))

    for paragraph in second_seq:
        max_p_len = max(max_p_len , len(paragraph))
        for sent in paragraph:
            max_sent_len = max(max_sent_len , len(sent))

    first = np.zeros((2 * len(paragraphs), max_p_len,max_sent_len), dtype=np.int32)
    second = np.zeros((2 * len(paragraphs), max_p_len,max_sent_len), dtype=np.int32)

    labels = np.zeros((2 * len(paragraphs), 2), dtype=np.float32)
    labels[:len(paragraphs), 0] = 1
    labels[len(paragraphs):, 1] = 1

    for i, p in enumerate(first_seq):
        for j, sent in enumerate(p):
            for k,word in enumerate(sent):
                if word in stoi:
                    first[i, j, k] = stoi[word]
                else:
                    first[i, j, k] = stoi['**BLANK**']

    for i, p in enumerate(second_seq):
        for j, sent in enumerate(p):
            for k,word in enumerate(sent):
                if word in stoi:
                    second[i, j, k] = stoi[word]
                else:
                    second[i, j, k] = stoi['**BLANK**']

    first[len(paragraphs):, :] = second[:len(paragraphs), :]
    second[len(paragraphs):, :] = first[:len(paragraphs), :]

    permutation = np.random.permutation(2 * len(paragraphs))

    first = first[permutation]
    second = second[permutation]
    labels = labels[permutation]

    return first, second, labels


def get_paragraphs_list(stoi, max_len=512 , min_len=10 , min_sent=3 ,paragraph_id=None, books=None, tags=None,
                            num_sequential=2, Paragraph_Object=True):
    paragraphs = get_paragraphs(paragraph_id=paragraph_id, books=books,
                                tags=tags, num_sequential=num_sequential, paragraph_object=Paragraph_Object)

    paragraphs = filter(paragraphs, max_len, min_len, min_sent)

    ret=[]

    for tuple in paragraphs:
        tuple_indexed=[]
        for p in tuple:
            p_indexed=[]
            sents = p.text(format="sentences")
            for sent in sents:
                sent_indexed=[]
                for word in sent:
                    if word in stoi:
                        sent_indexed.append(stoi[word])
                    else:
                        sent_indexed.append(stoi['**BLANK**'])

                p_indexed.append(sent_indexed)
            tuple_indexed.append(p_indexed)
        ret.append(tuple_indexed)

    shuffle(ret)

    print(ret[0])


    return ret

def get_paragraphs_as_text(max_len=512 , min_len=10 , min_sent=3 , paragraph_id=None, books=None, tags=None,
                            num_sequential=2, Paragraph_Object=True):
    paragraphs = get_paragraphs(paragraph_id=paragraph_id, books=books,
                                tags=tags, num_sequential=num_sequential, paragraph_object=Paragraph_Object)

    paragraphs = filter(paragraphs, max_len, min_len, min_sent)

    ret = []

    for x,y in paragraphs:
        ret.append((x.text("text") , y.text("text")))
















