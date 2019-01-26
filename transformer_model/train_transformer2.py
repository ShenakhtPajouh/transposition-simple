import tensorflow as tf
from transformer_batch_generator2 import batch_generator
from transformer_transposition2 import transformer
import numpy as np
import pickle

def train(train_files , val_files , batch_size ,max_sent_num , embedding_len,
          epochs ,  learning_rate , lr_decay):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #defining the model

    model = transformer( "transformer2" , embedding_len  , max_sent_num , batch_size , train = True)

    train_batches = batch_generator(train_files, batch_size=batch_size , max_sent_num=max_sent_num ,
                                    embedding_len = embedding_len , epochs = epochs )

    with tf.Session(config=config) as sess:
        #loading pretrained bert weights
        #print ([v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='bert')])


        tmp = learning_rate
        lr = tf.Variable(0.0, trainable=False)
        lr_new_value = tf.placeholder(tf.float32, [])
        lr_update = tf.assign(lr, lr_new_value)


        global_step = tf.Variable(0, trainable=False)
        params = model.variables()
        gradients = tf.gradients(model.loss , params)

        # declaring the optimizer
        optimizer = tf.train.AdamOptimizer(lr)
        train_opt = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)

        sess.run(tf.global_variables_initializer())

        total_loss=0
        count=0

        losses = []
        acc = []

        for batch_num , (first_num , second_num , inputs  , labels) in enumerate(train_batches):
            count+=1
            _, loss = sess.run([train_opt , model.loss] ,
                               feed_dict={model.first_num_sents: first_num,
                                          model.second_num_sents: second_num,
                                          model.inputs : inputs,
                                          model.target: labels})

            total_loss+=loss

            if (batch_num%100==0):
                losses.append(total_loss/count)
                print("Step: "+str(batch_num)+": loss=" + str(total_loss/count))

            if (batch_num%5000==0):
                validation_batches = batch_generator(val_files, batch_size=batch_size , max_sent_num=max_sent_num ,
                                        embedding_len = embedding_len , epochs = 1)
                val_count = 0
                val_correct = 0
                for first_num , second_num , inputs , labels in validation_batches:
                    val_count+=1
                    loss , logits = sess.run([model.loss , model.final_logits] ,
                                       feed_dict={model.first_num_sents: first_num,
                                                  model.second_num_sents: second_num,
                                                  model.inputs : inputs,
                                                  model.target: labels})

                    ans = np.argmax(logits , axis=1)
                    for i in range (len(ans)):
                        if (labels[i][ans[i]]==1):
                            val_correct+=1

                acc.append(val_correct/(batch_size*val_count))
                print ("Step: "+str(batch_num)+":validation accuracy = "+ str(val_correct/(batch_size*val_count)))
                total_loss = 0

                tmp = tmp * lr_decay
                sess.run(lr_update, feed_dict={lr_new_value: tmp})

                with open("transformer2_losses.pkl" , 'wb') as pkl:
                    pickle.dump(losses , pkl)

                with open("transformer2_val_acc.pkl" , 'wb') as pkl:
                    pickle.dump(acc , pkl)

        saver2 = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
        saver2.save(sess, './bert_transformer_2.ckpt')






if __name__=='__main__':
    train_files = ['paragraph_pairs_encoded'+str(i)+'.pkl' for i in range (1,20)]
    val_files = ['paragraph_pairs_encoded20.pkl']
    train(train_files , val_files , batch_size=64  , max_sent_num = 37, embedding_len = 768,
          epochs=100 ,  learning_rate=0.01 , lr_decay=0.9)




