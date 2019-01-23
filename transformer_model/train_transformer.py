import tensorflow as tf
from transformer_model.transformer_batch_generator import batch_generator
from transformer_model.transformer_transposition import trasformer
import utils
import numpy as np
from random import shuffle

def train(dataset , batch_size , max_sent_len , max_sent_num , embedding_len,
          bert_vocab_path , bert_config_path , bert_checkpoint ,
          epochs ,  learning_rate , lr_decay , validation_split):

    validation = dataset[:int(len(dataset)*validation_split)]
    train = paragraphs[int(len(dataset)*validation_split):]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #defining the model
    with tf.device('/gpu:0'):
        model = trasformer( "transformer1" , embedding_len , bert_config_path , max_sent_num , batch_size)

    train_batches = batch_generator(train, batch_size=64, max_sent_len=max_sent_len,
                              bert_vocab_path=bert_vocab_path)

    validation_batches =  batch_generator(validation, batch_size=64, max_sent_len=max_sent_len,
                              bert_vocab_path=bert_vocab_path)

    with tf.session(config=config) as sess:
        #loading pretrained bert weights
        saver = tf.train.Saver()
        saver.restore(sess, bert_checkpoint)

        tmp = learning_rate
        lr = tf.Variable(0.0, trainable=False)
        lr_new_value = tf.placeholder(tf.float32, [])
        lr_update = tf.assign(lr, lr_new_value)

        # clipping gradients by norm 5
        global_step = tf.Variable(0, trainable=False)
        params = model.variables()
        gradients = tf.gradients(model.loss , params)

        # declaring the optimizer
        optimizer = tf.train.AdamOptimizer(lr)
        train_opt = optimizer.apply_gradients(zip(gradients, params), global_step=global_step)

        for epoch in range (epochs):
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

                total_loss+=loss/batch_size

                if (batch_num%100==0):
                    print("Epoch "+str(epoch)+":batch "+str(batch_num)+": loss=" + str(total_loss/count))

            validation_loss=0
            for batch_num , (first_num , second_num , indices , inputs , mask , labels) in enumerate(validation_batches):
                count+=1
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
                print ("Epoch "+str(epoch)+":validation accuracy = "+ str(correct/batch_size*count))

            tmp = tmp * lr_decay
            sess.run(lr_update, feed_dict={lr_new_value: tmp})


        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
        saver.save(sess, './bert_transformer_1.ckpt')







if __name__=='__main__':
    paragraphs = utils.get_sents_as_text_list(max_len=512 , min_len=20 , min_sent=3 , tags=[[0,1,2]])

    #adding labels and reversed order of pairs to the data
    all_paragraphs = [(x,y,1) for x,y in paragraphs]
    all_paragraphs = all_paragraphs+[(y,x,0) for x,y in paragraphs]
    shuffle (all_paragraphs)

    max_sent_len = -1
    max_sent_num = -1
    for x,y in paragraphs:
        max_sent_num = min (max_sent_num , len(x) , len(y))
        for sent in x:
            max_sent_len = max(len(sent) , max_sent_len)
        for sent in y:
            max_sent_len = max(len(sent), max_sent_len)




