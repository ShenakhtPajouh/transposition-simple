#training transformer_transposition

import tensorflow as tf
from transformer_batch_generator import batch_generator
from transformer_transposition import transformer
import numpy as np
from random import shuffle
import pickle

def train(dataset , batch_size , max_sent_len , max_sent_num , embedding_len,
          bert_vocab_path , bert_config_path , bert_checkpoint ,
          epochs ,  learning_rate , lr_decay , validation_split):

    validation = dataset[:int(len(dataset)*validation_split)]
    train = dataset[int(len(dataset)*validation_split):]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #defining the model

    model = transformer( "transformer1" , embedding_len , bert_config_path , max_sent_num , batch_size , train = True)

    train_batches = batch_generator(train, batch_size=batch_size , max_sent_len=max_sent_len,
                              bert_vocab_path=bert_vocab_path , epochs=epochs)

    validation_batches =  batch_generator(validation, batch_size= batch_size, max_sent_len=max_sent_len,
                              bert_vocab_path=bert_vocab_path)

    with tf.Session(config=config) as sess:
        #loading pretrained bert weights

        saver = tf.train.Saver({v.name: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='bert')})
        saver.restore(sess, bert_checkpoint)
        #used for decreasing the learning rate
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

        #used for computing loss
        total_loss=0
        count=0

        for batch_num , (first_num , second_num , indices , inputs , mask , labels) in enumerate(train_batches):
            count+=1
            _, loss = sess.run([train_opt , model.loss] ,
                               feed_dict={model.first_num_sents: first_num,
                                          model.second_num_sents: second_num,
                                          model.indices : indices,
                                          model.inputs : inputs,
                                          model.mask :mask,
                                          model.target: labels})

            total_loss+=loss

            #computing loss each 100 steps
            if (batch_num%100==0):
                print("Step: "+str(batch_num)+": loss=" + str(total_loss/count))

            #computing validation accuracy and updating learning rate each 5000 steps
            if (batch_num%5000==0):
                val_count = 0
                for batch_num , (first_num , second_num , indices , inputs , mask , labels) in enumerate(validation_batches):
                    val_count+=1
                    loss , logits = sess.run([model.loss , model.final_logits] ,
                                       feed_dict={model.first_num_sents: first_num,
                                                  model.second_num_sents: second_num,
                                                  model.indices : indices,
                                                  model.inputs : inputs,
                                                  model.mask :mask,
                                                  model.target: labels})

                    ans = np.argmax(logits , axis=1)
                    correct=0
                    for i in range (len(ans)):
                        if (labels[i][ans[i]]==1):
                            correct+=1
                print (val_count)
                print (batch_size)
                print ("Step: "+batch_num+":validation accuracy = "+ str(correct/(batch_size*val_count)))

                loss = 0
                count = 0

                tmp = tmp * lr_decay
                sess.run(lr_update, feed_dict={lr_new_value: tmp})


    saver2 = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
    saver2.save(sess, './bert_transformer_1.ckpt')






if __name__=='__main__':
    with open ("paragraph_sents.pkl" , "rb") as pkl:
       paragraphs = pickle.load(pkl)

    #adding reverse order to the dataset
    all_paragraphs = [(x,y,1) for x,y in paragraphs]
    all_paragraphs = all_paragraphs+[(y,x,0) for x,y in paragraphs]
    shuffle (all_paragraphs)


    train(all_paragraphs[:1000] , 32 , 300 , 37 , 768,
          "/mnt/storage1/Data/BERT/uncased_L-12_H-768_A-12*/vocab.txt" , "/mnt/storage1/Data/BERT/uncased_L-12_H-768_A-12*/bert_config.json" , "/mnt/storage1/Data/BERT/new/bert_vars.ckpt" ,
          100 ,  0.01 , 0.01 , 0.1)




