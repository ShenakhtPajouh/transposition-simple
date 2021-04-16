import numpy as np
import tensorflow as tf
import pickle
import time


def f(a, b, a_len, b_len, batch_size, max_len, dim):
    c = np.zeros((batch_size, 2 * max_len, dim), dtype=np.float32)
    mask = np.zeros((batch_size, 2 * max_len), dtype=np.float32)

    for i in range(batch_size):
        c[i][:a_len[i]] = a[i][:a_len[i]]
        c[i][a_len[i]:a_len[i] + b_len[i]] = b[i][:b_len[i]]
        mask[i][:a_len[i] + b_len[i]] = np.tile([1], a_len[i] + b_len[i])

    return c, mask


def validation_generator(batch_size):
    pairs_encoded = []
    for i in range(3):
        path = '../paragraph_pairs_encoded' + str(19 + i) + '.pkl'
        with open(path, 'rb') as pkl:
            pairs_encoded.append(pickle.load(pkl))

    with open('../keys_encoded9.pkl', 'rb') as pkl:
        keys = pickle.load(pkl)
    keys = np.float32(np.append(keys, keys, axis=0))

    with open('../mask_encoded9.pkl', 'rb') as pkl:
        keys_mask = pickle.load(pkl)
    keys_mask = np.float32(np.append(keys_mask, keys_mask, axis=0))

    max_len = 37
    dim = 768
    encoded = np.empty(shape=[0, 2 * max_len, dim], dtype=np.float32)
    masks = np.empty(shape=[0, 2 * max_len], dtype=np.float32)
    labels = np.empty(shape=[
        0,
    ], dtype=np.int32)

    for i in range(2):
        for j in range(3):
            e, mask = f(pairs_encoded[j][i], pairs_encoded[j][i ^ 1],
                        pairs_encoded[j][2 + i], pairs_encoded[j][2 + (i ^ 1)],
                        len(pairs_encoded[j][3]), max_len, dim)
            encoded = np.append(encoded, e, axis=0)
            masks = np.append(masks, mask, axis=0)
            labels = np.append(labels, np.tile([i ^ 1],
                                               len(pairs_encoded[j][3])))

    del pairs_encoded

    n = len(labels)
    pi = np.random.permutation(n)
    encoded = encoded[pi]
    masks = masks[pi]
    keys = keys[pi]
    keys_mask = keys_mask[pi]
    labels = labels[pi]

    for i in range(0, n, batch_size):
        if i + batch_size > n:
            break
        yield (encoded[i:i + batch_size], masks[i:i + batch_size],
               keys[i:i + batch_size], keys_mask[i:i + batch_size],
               labels[i:i + batch_size])

    del labels
    del keys_mask
    del keys
    del masks
    del encoded


def batch_generator(batch_size):
    for epoch in range(10):
        for b in range(9):
            pairs_encoded = []
            for i in range(2):
                path = '../paragraph_pairs_encoded' + str(2 * b + 1 +
                                                          i) + '.pkl'
                with open(path, 'rb') as pkl:
                    pairs_encoded.append(pickle.load(pkl))

            path = '../keys_encoded' + str(b) + '.pkl'
            with open(path, 'rb') as pkl:
                keys = pickle.load(pkl)
            keys = np.float32(np.append(keys, keys, axis=0))

            path = '../mask_encoded' + str(b) + '.pkl'
            with open(path, 'rb') as pkl:
                keys_mask = pickle.load(pkl)
            keys_mask = np.float32(np.append(keys_mask, keys_mask, axis=0))

            max_len = 37
            dim = 768
            encoded = np.empty(shape=[0, 2 * max_len, dim], dtype=np.float32)
            masks = np.empty(shape=[0, 2 * max_len], dtype=np.float32)
            labels = np.empty(shape=[
                0,
            ], dtype=np.int32)

            for i in range(2):
                for j in range(2):
                    e, mask = f(pairs_encoded[j][i], pairs_encoded[j][i ^ 1],
                                pairs_encoded[j][2 + i],
                                pairs_encoded[j][2 + (i ^ 1)],
                                len(pairs_encoded[j][3]), max_len, dim)
                    encoded = np.append(encoded, e, axis=0)
                    masks = np.append(masks, mask, axis=0)
                    labels = np.append(
                        labels, np.tile([i ^ 1], len(pairs_encoded[j][3])))

            del pairs_encoded

            n = len(labels)
            pi = np.random.permutation(n)
            encoded = encoded[pi]
            masks = masks[pi]
            keys = keys[pi]
            keys_mask = keys_mask[pi]
            labels = labels[pi]

            for i in range(0, n, batch_size):
                if i + batch_size > n:
                    break
                yield (encoded[i:i + batch_size], masks[i:i + batch_size],
                       keys[i:i + batch_size], keys_mask[i:i + batch_size],
                       labels[i:i + batch_size])

            del labels
            del keys_mask
            del keys
            del masks
            del encoded
