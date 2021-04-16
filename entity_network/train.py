import tensorflow as tf
from tensorflow.keras import losses
import numpy as np
from random import shuffle
from batch_generator import *
from entity_main_model import entity_main_model
import matplotlib.pyplot as plt
import os
import pickle
import time

tf.enable_eager_execution()


def reshape(a, i, rep):
    a = np.expand_dims(a, axis=i)
    return np.repeat(a, axis=i, repeats=rep)


def one_hot(a):
    b = np.zeros((len(a), 2))
    b[np.arange(len(a)), a] = 1
    return reshape(b, 1, 20)


def get_loss(inputs, model, y_true, mask):
    y_pred = model(inputs)
    return tf.reduce_mean(
        tf.multiply(losses.categorical_crossentropy(one_hot(y_true), y_pred),
                    mask))


def grad(inputs, model, labels, mask):
    with tf.GradientTape() as tape:
        loss = get_loss(inputs, model, labels, mask)
    return loss, tape.gradient(loss, model.variables)


def train_plot(train_loss_results, train_accuracy_results):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Train Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].plot(train_accuracy_results)
    axes[1].set_xlabel("Epoch", fontsize=14)


def train(model,
          learning_rate,
          num_epochs,
          save_path,
          batch_size,
          train_steps=50,
          validation_steps=1000):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tfe = tf.contrib.eager
    train_loss_results = []
    train_accuracy_results = []
    validation_accuracy_results = []
    step = 0
    loss_avg = tfe.metrics.Mean()
    train_accuracy = tfe.metrics.Accuracy()
    batchgenerator = batch_generator(batch_size)

    for batch in batchgenerator:
        #inputs = (seq_encoded, masks, keys_vals)
        inputs = batch[:3]
        keys_mask = batch[3]
        labels = batch[4]
        outputs = model(inputs)
        loss, grads = grad(inputs, model, labels, keys_mask)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  tf.train.get_or_create_global_step())
        loss_avg(loss)
        train_accuracy(
            np.argmax(np.apply_along_axis(np.bincount, 1,
                                          np.argmax(outputs.numpy(), axis=2)),
                      axis=1), labels)
        step += 1

        if step % train_steps == 0:
            train_loss_results.append(loss_avg.result().numpy())
            train_accuracy_results.append(train_accuracy.result().numpy())
            print("Step " + str(step) + ": " + "Train Loss= " +
                  str(loss_avg.result().numpy()) + ", Train Accuracy= " +
                  str(train_accuracy.result().numpy()))
            loss_avg = tfe.metrics.Mean()
            train_accuracy = tfe.metrics.Accuracy()

            if step % validation_steps == 0:
                validation_accuracy = tfe.metrics.Accuracy()
                validationgenerator = validation_generator(batch_size)
                for validation in validationgenerator:
                    x = validation[:3]
                    y = validation[4]
                    print(model(x).numpy().shape)
                    validation_accuracy(
                        np.argmax(np.apply_along_axis(
                            np.bincount, 1, np.argmax(model(x).numpy(),
                                                      axis=2)),
                                  axis=1), y)

                validation_accuracy_results.append(
                    validation_accuracy.result().numpy())
                print("Validation Accuracy= " +
                      str(validation_accuracy.result().numpy()))

                checkpoint_dir = save_path
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_dir = os.path.join(checkpoint_dir, 'ckpt')
                tfe.Saver(model.variables).save(checkpoint_dir)

                with open('train_accuracy_results.pkl', 'wb') as pkl:
                    pickle.dump(train_accuracy_results, pkl)

                with open('train_loss_results.pkl', 'wb') as pkl:
                    pickle.dump(train_loss_results, pkl)

                with open('validation_accuracy_results.pkl', 'wb') as pkl:
                    pickle.dump(validation_accuracy_results, pkl)

    train_plot(train_loss_results, train_accuracy_results)


#     validation_plot(validation_accuracy_results)

if __name__ == "__main__":
    model = entity_main_model("EntityModel", 20, 768, 768, 80, 0)
    train(model, 0.001, 1, './ModelVariables', 64)
