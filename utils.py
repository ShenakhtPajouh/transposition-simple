import numpy as np
import pickle
from remote.API import get_paragraphs
import numpy as np


def extract_embedding_from_dict(dict_path , vocab_size , embedding_len , vectors_path , stoi_path , itos_path):
    """
    :param dict_path: path of the dictionary of word to embedding
    :param vocab_size: size of embedding vocabulary
    :param embedding_len: length of embedding vector
    """

    # an np array of shape [vocab_size , embedding_len] containing embedding vectors
    vectors = np.zeros ((vocab_size, embedding_len) , dtype=np.float32)
    #a dictionary that maps words to their corresponding index in vectors
    stoi = dict()
    #a list in which index of each word is its index in vectors
    itos = []

    i=0
    with open (dict_path , 'r' , encoding= 'UTF-8') as infile:
        for line in infile.readlines():
            row = line.strip().split(' ')
            itos.append(row[0])
            listt = [float(j) for j in row[1:]]
            stoi[row[0]]=i
            vectors[i]=np.array(listt)
            i+=1

    with open (vectors_path, 'wb') as pkl:
        pickle.dump(vectors , pkl)
    with open (stoi_path , 'wb') as pkl:
        pickle.dump(stoi , pkl)
    with open (itos_path , 'wb') as pkl:
        pickle.dump(itos , pkl)


def get_from_file (path):
    """
    :param path: path of the file
    :return: object read from file
    """
    with open(path, 'rb') as pkl:
        ret = pickle.load(pkl)

    return ret

def get_paragraphs_as_words(stoi , paragraph_id=None, books=None, tags=None,
                 num_sequential=2, Paragraph_Object=False):
    """
    :return: a triple of np arrays: first: embedding of words of first paragraphs per row
                                    second: embedding of words of second paragraphs per row
                                    labels: whether the first paragraph has happend before the second paragraph
    """

    paragraphs = get_paragraphs(paragraph_id = paragraph_id, books=books,
                                tags=tags, num_sequential=num_sequential, Paragraph_Object=Paragraph_Object)

    max_len=-1

    first_seq = []
    second_seq = []

    for x,y in paragraphs:
        for sent in x:
            for word in sent:
                first_seq.append(word)
        for sent in y:
            for word in sent:
                second_seq.append(word)


    for seq in first_seq:
        max_len = max(max_len , len(seq))

    for seq in second_seq:
        max_len = max (max_len , len(seq))

    first = np.zeros((2*len(paragraphs),max_len),dtype=np.int32)
    second = np.zeros((2*len(paragraphs), max_len), dtype=np.int32)

    labels = np.zeros((2*len(paragraphs),2) , dtype=np.float64)
    labels[:len(paragraphs),0]=1
    labels[len(paragraphs):,1] = 1

    for i ,seq in enumerate(first_seq):
        for j , word in enumerate(seq):
            first[i,j]=stoi[word]

    for i ,seq in enumerate(second_seq):
        for j , word in enumerate(seq):
            second[i,j]=stoi[word]

    first[len(paragraphs):,:]=second[:len(paragraphs),:]
    second[len(paragraphs):, :] = first[:len(paragraphs), :]

    permutation = np.random.permutation(2*len(paragraphs))

    first = first[permutation]
    second = second[permutation]
    labels = labels[permutation]

    return first , second , labels









