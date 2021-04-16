import tensorflow as tf
import matplotlib.pyplot as plt
import utils
from Simpler_Models import main_model3


def train(embedding_table, batch_size, epochs, learning_rate, hidden_len, data,
          validation_split, save_path):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.keras.backend.set_session(session)

    with tf.device('/gpu:0'):
        model = main_model3.main_model('lstm-glucnn', embedding_table,
                                       hidden_len)

    adam = tf.keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(save_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(x=[data[0], data[1], data[3], data[4]],
                        y=data[2],
                        batch_size=batch_size,
                        validation_split=validation_split,
                        epochs=epochs,
                        callbacks=[cp_callback])

    return history


def plot(name, history):
    plt.figure(figsize=(16, 10))
    val = plt.plot(history.epoch,
                   history.history['val_categorical_crossentropy'],
                   '--',
                   label=name + ' Val')
    plt.plot(history.epoch,
             history.history['categorical_crossentropy'],
             color=val[0].get_color(),
             label=name + ' Train')
    plt.xlabel('Epochs')
    plt.ylabel('categorical_crossentropy'.replace('_', ' ').title())
    plt.legend()
    plt.xlim([0, max(history.epoch)])


if __name__ == '__main__':
    stoi = utils.get_from_file('stoi.pkl')
    vectors = utils.get_from_file('vectors.pkl')

    first = utils.get_from_file('first.pkl')
    second = utils.get_from_file('second.pkl')
    labels = utils.get_from_file('labels.pkl')
    indices = utils.get_from_file('indices.pkl')

    train_data = (first[:(first.shape[0] // 10) * 9, :],
                  second[:(second.shape[0] // 10) * 9, :],
                  labels[:(labels.shape[0] // 10) * 9, :],
                  indices[0][:(labels.shape[0] // 10) * 9, :],
                  indices[1][:(labels.shape[0] // 10) * 9, :])

    test_data = (first[(first.shape[0] // 10) * 9:, :],
                 second[(second.shape[0] // 10) * 9:, :],
                 labels[(labels.shape[0] // 10) * 9:, :],
                 indices[0][:(labels.shape[0] // 10) * 9, :],
                 indices[1][:(labels.shape[0] // 10) * 9, :])

    history = train(embedding_table=vectors,
                    batch_size=64,
                    epochs=50,
                    learning_rate=0.05,
                    hidden_len=200,
                    validation_split=0.05,
                    data=train_data,
                    save_path='./model2.ckpt')

    plot('LSTM+GLUCNN', history)
