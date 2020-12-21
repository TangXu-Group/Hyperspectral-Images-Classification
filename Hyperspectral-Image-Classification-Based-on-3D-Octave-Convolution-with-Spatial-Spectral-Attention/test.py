from model import *
import numpy as np

input_data = tf.placeholder(tf.float32, [None, 13, 13, 103])
input_labels = tf.placeholder(tf.float32, [None, 9])
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

data_test = np.load("./data_test.npy")
labels_test = np.load("./labels_test.npy")
print("data_test" + str(data_test.shape))
print("labels_test" + str(labels_test.shape))

logit_spatial, logit_spectral, logit, predict = network(input_data, weights, keep_prob, is_training)

test_number = data_test.shape[0]
test_number_int = test_number // 256
test_batchnumber = test_number_int + 1

predict_labels = np.zeros((labels_test.shape[0], 9))

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,
                  "./Weights_Pavia/weights_pavia.ckpt")
    for i in range(test_batchnumber):
        start_test = i * 256
        end_test = min(start_test + 256, test_number)
        pre_test = sess.run(predict, feed_dict={input_data: data_test[start_test:end_test],
                                                 input_labels: labels_test[start_test:end_test],
                                                 keep_prob: 1.0,
                                                 is_training: False})

        predict_labels[start_test:end_test, ...] = pre_test

    print("predict_labels" + str(predict_labels.shape))

    matrix = np.zeros((9, 9))

    for j in range(len(predict_labels)):
        o = predict_labels[j, ...]
        p = np.argmax(o)
        q = labels_test[j, ...]
        r = np.argmax(q)
        matrix[p, r] += 1

    OA = np.sum(np.trace(matrix)) / np.sum(matrix)
    print('OA:', OA)

    ac_list = np.zeros((9))

    for k in range(len(matrix)):
        ac_k = matrix[k, k] / sum(matrix[:, k])
        ac_list[k] = ac_k
        print("ac" + str(k+1) +":"+ str(ac_k))

    AA = np.mean(ac_list)
    print("AA:" + str(AA))

    mm = 0
    for l in range(matrix.shape[0]):
        mm += np.sum(matrix[l]) * np.sum(matrix[:, l])
    pe = mm / (np.sum(matrix) * np.sum(matrix))
    pa = np.trace(matrix) / np.sum(matrix)
    kappa = (pa - pe) / (1 - pe)
    print("kappa" + str(kappa))




