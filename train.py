import tensorflow as tf
import utils
import main_model


def train(embedding_table, batch_size, epochs, learning_rate, hidden_len, encoder_dropout, classifier_dropout, data,
          save_path):
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

    model.fit(dataset, epochs=epochs, steps_per_epoch=5, callbacks=[cp_callback])


if __name__ == '__main__':
    stoi = utils.get_from_file('stoi.pkl')
    vectors = utils.get_from_file('vectors.pkl')

    data = utils.get_from_file('data.pkl')

    train_data = (data[0][:(data[0].shape[0] // 10) * 7, :], data[1][:(data[1].shape[0] // 10) * 7, :],
                  data[2][:(data[2].shape[0] // 10) * 7, :])

    train(embedding_table=vectors, batch_size=64, epochs=10, learning_rate=0.001, hidden_len=200,
          encoder_dropout=0.5, classifier_dropout=0.2, data=train_data, save_path='./model.ckpt')

