import tensorflow as tf
from transformer_batch_generator2 import batch_generator
from transformer_transposition2 import transformer
import numpy as np


def test(files, batch_size, max_sent_num, embedding_len, model_path):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #defining the model

    model = transformer("transformer2",
                        embedding_len,
                        max_sent_num,
                        batch_size,
                        train=True)

    test_batches = batch_generator(files,
                                   batch_size=batch_size,
                                   max_sent_num=max_sent_num,
                                   embedding_len=embedding_len,
                                   epochs=1)

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)

        total_loss = 0
        count = 0
        correct = 0

        for batch_num, (first_num, second_num, inputs,
                        labels) in enumerate(test_batches):
            count += 1
            loss, logits = sess.run(
                [model.loss, model.final_logits],
                feed_dict={
                    model.first_num_sents: first_num,
                    model.second_num_sents: second_num,
                    model.inputs: inputs,
                    model.target: labels
                })

            total_loss += loss

        ans = np.argmax(logits, axis=1)
        for i in range(len(ans)):
            if (labels[i][ans[i]] == 1):
                correct += 1

        print(":accuracy = " + str(correct / (batch_size * count)))
        print("loss = " + str(total_loss / count))


if __name__ == '__main__':
    files = ['paragraph_pairs_encoded' + str(i) + '.pkl' for i in range(1, 20)]
    test(files,
         batch_size=64,
         max_sent_num=37,
         embedding_len=768,
         model_path="./bert_transformer_2adam001.ckpt")
