import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import main_model


def train(embedding_table, batch_size, epochs, learning_rate, hidden_len, encoder_dropout, classifier_dropout, data, val_data ,
           validation_split , save_path):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)

    with tf.device('/gpu:0'):
        model = main_model.main_model('lstm-fc', embedding_table, hidden_len, encoder_dropout, classifier_dropout)

    adam = tf.keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    dataset_12 = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
    dataset_label = tf.data.Dataset.from_tensor_slices(data[2])
    dataset = tf.data.Dataset.zip((dataset_12, dataset_label)).batch(batch_size, drop_remainder=True)

    val_dataset_12 = tf.data.Dataset.from_tensor_slices((val_data[0], val_data[1]))
    val_dataset_label = tf.data.Dataset.from_tensor_slices(val_data[2])
    val_dataset = tf.data.Dataset.zip((val_dataset_12, val_dataset_label)).batch(batch_size , drop_remainder=True)

    #history = model.fit(x = [data[0],data[1]] , y = data[2] , batch_size = batch_size, validation_split = validation_split ,epochs=epochs, callbacks=[cp_callback])

    history = model.fit(dataset, epochs=10, steps_per_epoch=30,validation_data=val_dataset,validation_steps=3,callbacks=[cp_callback])
    return history


def plot (name , history):
        plt.figure(figsize=(16, 10))
        val = plt.plot(history.epoch, history.history['val_binary_crossentropy'],
                       '--', label=name+' Val')
        plt.plot(history.epoch, history.history['binary_crossentropy'], color=val[0].get_color(),
                 label=name+' Train')
        plt.xlabel('Epochs')
        plt.ylabel('binary_crossentropy'.replace('_', ' ').title())
        plt.legend()
        plt.xlim([0, max(history.epoch)])


if __name__ == '__main__':
    stoi = utils.get_from_file('stoi.pkl')
    vectors = utils.get_from_file('vectors.pkl')

    first = utils.get_from_file('first.pkl')
    second = utils.get_from_file('second.pkl')
    labels = utils.get_from_file('labels.pkl')

    data = utils.make_dataset_with_indices(stoi , tags=[1,2])

    print(data[3].shape)
    #train_data = (first[:(first.shape[0] // 10) * 9, :], second[:(second.shape[0] // 10) * 9, :],
    #             labels[:(labels.shape[0] // 10) * 9, :])
    #test_data = (first[(first.shape[0] // 10) * 9:, :],
    #           second[(second.shape[0] // 10) * 9:, :],
    #           labels[(labels.shape[0] // 10) * 9:, :])



    history = train(embedding_table=vectors, batch_size=64, epochs=50, learning_rate=0.001, hidden_len=200,
          encoder_dropout=0.5, classifier_dropout=0.2 , validation_split = 0.05, data=train_data, save_path='./model.ckpt')


    #plot ('LSTM + FC' , history)


