import time
import numpy as np
import os
import tensorflow.compat.v1 as tf
import matplotlib.transforms as mtransforms
tf.disable_v2_behavior()
from Model.cnn import Model
import matplotlib.pyplot as plt
import datetime


def evaluate_model():
    model = Model()
    model_path = './Model/model_trained/model_save.ckpt'
    trains = np.load('./test_data/2chip/Trains.npy')
    labels = np.load('./test_data/2chip/Labels.npy')
    xs = tf.placeholder(tf.float32, [None, 250, 250, 1])
    x_image = tf.reshape(xs, [-1, 250, 250, 1])
    y_pre = model.inference(x_image)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)  # 加载变量值
        number = np.random.randint(len(trains))
        train = trains[number][np.newaxis, :, :, np.newaxis]
        label = labels[number]
        pre = sess.run(y_pre, feed_dict={xs: train}).reshape(250, 250)
        error = np.abs(pre - label) / label.ptp()
    return pre, label, error


def main():
    pre, label, error = evaluate_model()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.imshow(pre, cmap=plt.cm.get_cmap('jet'))
    ax2.imshow(label, cmap=plt.cm.get_cmap('jet'))
    axes3 = ax3.imshow(error, cmap=plt.cm.get_cmap('jet'))
    axpos = ax3.get_position()
    caxpos = mtransforms.Bbox.from_extents(
        axpos.x1 + 0.01,
        axpos.y0,
        axpos.x1 + 0.01 + 0.01,
        axpos.y1
    )
    cax = ax3.figure.add_axes(caxpos)
    cbar = fig.colorbar(axes3, cax=cax)
    plt.show()



if __name__ == '__main__':
    main()
