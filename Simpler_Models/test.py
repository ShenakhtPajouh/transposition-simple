import tensorflow as tf
import numpy as np
import sent_encoder
tf.enable_eager_execution()

if __name__ == '__main__':
    tensor = [[[1, 2], [3, 4]],[[5, 6],[7,8]]]
    mask = np.array([[[0,0],[0,1]],[[1,0],[10,10]]])
    print (tf.gather_nd(tensor, mask))  # [[1, 2], [5, 6]]


