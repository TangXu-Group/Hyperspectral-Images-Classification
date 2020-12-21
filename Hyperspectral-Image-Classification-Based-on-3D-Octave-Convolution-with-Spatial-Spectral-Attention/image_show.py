from Code_Pavia_university.attention_pavia.model import *
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

input_data = tf.placeholder(tf.float32, [None, 13, 13, 103])
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

data_all = np.load("/home/MFB/PycharmProjects/Data_Use/DATA_pavia_university/data_all.npy")
hang_all = np.load("/home/MFB/PycharmProjects/Data_Use/DATA_pavia_university/hang.npy")
lie_all = np.load("/home/MFB/PycharmProjects/Data_Use/DATA_pavia_university/lie.npy")
print("data_all" + str(data_all.shape))
print("hang_all" + str(hang_all.shape))
print("lie_all" + str(lie_all.shape))

logit_spatial, logit_spectral, logit, predict = network(input_data, weights, keep_prob, is_training)

test_number = data_all.shape[0]
test_number_int = test_number // 256
test_batchnumber = test_number_int + 1

predict_labels = np.zeros((data_all.shape[0], 9))

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,
                  "/home/MFB/PycharmProjects/Code_Pavia_university/attention_pavia/Weights_Pavia/weights_pavia.ckpt")
    for i in range(test_batchnumber):
        start_test = i * 256
        end_test = min(start_test + 256, test_number)
        pre_test = sess.run(predict, feed_dict={input_data: data_all[start_test:end_test],
                                                 keep_prob: 1.0,
                                                 is_training: False})

        predict_labels[start_test:end_test, ...] = pre_test

    print("88888888888888888888888888888888888888888888888888")

result = np.zeros((610, 340))

for i in range(data_all.shape[0]):
    a = hang_all[i, ...]
    b = lie_all[i, ...]
    labels_range = predict_labels[i, ...]
    c = np.argmax(labels_range)
    result[a, b, ...] = c + 1
    print(result[a, b, ...])
print("result"+str(result.shape))

plt.imshow(result)
plt.show()