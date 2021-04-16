import pickle
import spacy
from spacy.lang.en import English
from bert_serving.client import BertClient
import numpy as np
import time
import string


def exist(words, word, i):
    if len(words) <= i:
        return False
    return words[i] == word


if __name__ == '__main__':

    with open('paragraph_pairs_text.pkl', 'rb') as pkl:
        paragraphs_texts = pickle.load(pkl)

    with open('paragraph_pairs_keys.pkl', 'rb') as pkl:
        paragraphs_keys = pickle.load(pkl)

    print('Here!')

    parser = spacy.load('en')
    tokenizer = English().Defaults.create_tokenizer(parser)
    bc = BertClient()
    printset = set(string.printable)

    slices = 10
    length = int(len(paragraphs_texts) / slices)

    for sli in range(slices - 4):
        sl = sli + 4
        start = time.time()
        embedded_keys = []
        begin = sl * length
        if sl == slices - 1:
            end = len(paragraphs_texts)
        else:
            end = (sl + 1) * length

        keys_path = 'keys_encoded' + str(sl) + '.pkl'
        mask_path = 'mask_encoded' + str(sl) + '.pkl'

        for ind, (pair_text, keys) in enumerate(
                zip(paragraphs_texts[begin:end], paragraphs_keys[begin:end])):
            if ind % 100 == 0:
                print(begin + ind)

            tokens1 = tokenizer(pair_text[0])
            tokens1 = [str(token).lower() for token in tokens1]
            tokens2 = tokenizer(pair_text[1])
            tokens2 = [str(token).lower() for token in tokens2]

            i = 0
            while i < len(tokens1):
                if set(tokens1[i]).issubset(printset) == False:
                    del tokens1[i]
                    i -= 1
                i += 1

            i = 0
            while i < len(tokens2):
                if set(tokens2[i]).issubset(printset) == False:
                    del tokens2[i]
                    i -= 1
                i += 1

            vecs1 = bc.encode([tokens1], is_tokenized=True)
            vecs2 = bc.encode([tokens2], is_tokenized=True)

            embedded_keys.append([])

            for word, indices in keys.items():
                cnt = 0
                e = np.zeros(768)

                for i in indices:
                    if exist(tokens1, word, i):
                        cnt += 1
                        e += vecs1[0][i + 1]
                    if exist(tokens2, word, i):
                        cnt += 1
                        e += vecs2[0][i + 1]

                if cnt > 0:
                    embedded_keys[-1].append(e / cnt)
                else:
                    embedded_keys[-1].append(e)

        keys = np.zeros((len(embedded_keys), 20, 768))
        mask = np.zeros((len(embedded_keys), 20))
        for i, paragraphs in enumerate(embedded_keys):
            for j, key in enumerate(paragraphs):
                keys[i][j] = key
                mask[i][j] = 1

        with open(keys_path, 'wb') as pkl:
            pickle.dump(keys, pkl)

        with open(mask_path, 'wb') as pkl:
            pickle.dump(mask, pkl)

        print('time = ' + str(time.time() - start))
