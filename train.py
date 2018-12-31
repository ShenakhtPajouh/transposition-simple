import tensorflow as tf
import numpy as np
import Simpler_Models
import utils

def train(embedding_table , batch_size , epochs, learning_rate ,  hidden_len , encoder_dropout, classifier_dropout , data , save_path):
    model = Simpler_Models.main_model('lstm-fc' ,embedding_table , hidden_len , encoder_dropout, classifier_dropout )

    model.compile(optimizer='adam',
                  loss='categorical_cross_entropy')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    model.fit(x=[data[0],data[1]] , y=data[2] , batch_size = batch_size , epochs=epochs , callbacks = [cp_callback])





if __name__=='__main__':
    stoi = utils.get_from_file('stoi.pkl')
    vectors = utils.get_from_file('vectors.pkl')
    data = utils.get_paragraphs_as_words(stoi)

    train(embedding_table=vectors,batch_size=64 , epochs = 10,learning_rate=0.001 , hidden_len=200 ,
          encoder_dropout=0.5 , classifier_dropout=0.2 , data = data , save_path='model.')

