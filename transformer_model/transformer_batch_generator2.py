import numpy as np
from random import shuffle
import pickle


def batch_generator(file_names, batch_size, max_sent_num , embedding_len , epochs):
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


    for epoch in range (epochs):
        print ("***EPOCH: "+str(epoch)+"***")
        shuffle(file_names)
        for name in file_names:
            with open (name , 'rb') as pkl:
                data = pickle.load(pkl)

            half_first = data[0]
            half_second = data[1]
            half_first_len = data[2]
            half_second_len = data[3]

            first = np.zeros((half_first.shape[0]*2 , max_sent_num , embedding_len))
            second = np.zeros((half_first.shape[0] * 2, max_sent_num , embedding_len))
            first_len = np.zeros((half_first.shape[0]*2) , np.int32)
            second_len = np.zeros((half_first.shape[0]*2) , np.int32)
            labels = np.zeros((half_first.shape[0]*2 , 2))

            labels[:half_first.shape[0] , 0] = 1
            labels[half_first.shape[0]:, 1] = 1

            first[:half_first.shape[0] , :half_first.shape[1]] = half_first
            first[half_first.shape[0]: , :half_first.shape[1]] = half_second

            second[:half_first.shape[0] , :half_first.shape[1]] = half_second
            second[half_first.shape[0]: , :half_first.shape[1]] = half_first

            first_len[:half_first.shape[0]] = half_first_len
            first_len[half_first.shape[0]:] = half_second_len

            second_len[:half_first.shape[0]] = half_second_len
            second_len[half_first.shape[0]:] = half_first_len

            permutation = np.random.permutation(first.shape[0])

            first = first[permutation]
            second = second[permutation]
            first_len = first_len[permutation]
            second_len = second_len[permutation]
            labels = labels[permutation]

            for i in range(0, first.shape[0]-first.shape[0]%batch_size , batch_size):
                first_len_batch = first_len[i:i + batch_size]
                second_len_batch = second_len[i:i + batch_size]
                labels_batch = labels[i:i + batch_size]


                inputs = np.zeros((batch_size , 2*max_sent_num+1 , embedding_len) , dtype=np.float32)

                for j in range (batch_size):

                    inputs[j][:first_len_batch[j]] = first[j][:first_len_batch[j]]
                    inputs[j][first_len_batch[j]+1:first_len_batch[j]+1+second_len_batch[j]] = second[j][:second_len_batch[j]]



                yield first_len_batch, second_len_batch, inputs , labels_batch