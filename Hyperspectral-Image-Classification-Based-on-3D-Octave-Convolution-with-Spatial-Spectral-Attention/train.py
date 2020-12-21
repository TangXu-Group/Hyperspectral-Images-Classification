from model import *
import numpy as np
import matplotlib.pyplot as plt

iteration = 100
batch_size = 16
learn_rate = 1e-4
tf.set_random_seed(9)

input_data = tf.placeholder(tf.float32, [None, 13, 13, 103])
input_labels = tf.placeholder(tf.float32, [None, 9])
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

data_train = np.load("./data_train.npy")
data_test = np.load("./data_test.npy")
labels_train = np.load("./labels_train.npy")
labels_test = np.load("./labels_test.npy")

logit_spatial, logit_spectral, logit, predict = network(input_data, weights, keep_prob, is_training)

with tf.name_scope('loss'):
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_spatial, labels=input_labels))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_spectral, labels=input_labels))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=input_labels))
    var = tf.trainable_variables()
    weights_decays = 0
    for item in var:
        weights_decay = tf.nn.l2_loss(item)
        weights_decays += weights_decay
    loss_op = loss1 + loss2 + loss3 + (0.0005 * weights_decays)

with tf.name_scope('optimizer'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss_op)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(input_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

train_number = data_train.shape[0]
batch_number_int = train_number // batch_size
batch_number = batch_number_int + 1

test_number = data_test.shape[0]
test_number_int = test_number // 256
test_batchnumber = test_number_int + 1

fig_loss = np.zeros([iteration])
fig_accuracy = np.zeros([iteration])
fig_test = np.zeros([iteration])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(iteration):
        # 打乱数据集
        idx = np.arange(0, data_train.shape[0])
        np.random.shuffle(idx)
        data_train_in = data_train[idx, ...]
        labels_train_in = labels_train[idx, ...]

        loss_list_train = np.zeros((batch_number))
        acc_list_train = np.zeros((batch_number))
        print("loss_list_train" + str(loss_list_train))
        print("loss_list_train_shape" + str(loss_list_train.shape))
        print("acc_list_train" + str(acc_list_train))
        print("acc_list_train_shape" + str(acc_list_train.shape))
        print("#######################################################")

        for batch in range(batch_number):
            start = batch * batch_size
            end = min(start + batch_size, train_number)

            loss, _ = sess.run([loss_op, optimizer],
                               feed_dict={input_data: data_train_in[start:end],
                                          input_labels: labels_train_in[start:end],
                                          keep_prob: 0.6,
                                          is_training: True})
            loss_list_train[batch, ...] = loss
            acc_train = sess.run(accuracy,
                                 feed_dict={input_data: data_train_in[start:end],
                                            input_labels: labels_train_in[start:end],
                                            keep_prob: 1.0,
                                            is_training: True})
            acc_list_train[batch] = acc_train
            print("epoch" + str(epoch) + "start" + str(start) +
                  "loss" + str(loss) + "acc_train" + str(acc_train))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("loss_list_train" + str(loss_list_train))
        print("acc_list_train" + str(acc_list_train))
        print("*************************************************************")
        fig_loss[epoch] = np.mean(loss_list_train)
        print("loss_average" + str(np.mean(loss_list_train)))
        fig_accuracy[epoch] = np.mean(acc_list_train)
        print("acc_average" + str(np.mean(acc_list_train)))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars

        idxen = np.arange(0, data_test.shape[0])
        np.random.shuffle(idxen)
        data_test_in = data_test[idxen, ...]
        labels_test_in = labels_test[idxen, ...]
        acc_list_test = np.zeros((test_batchnumber))

        if epoch > 50:
            for i in range(test_batchnumber):
                start_test = i * 256
                end_test = min(start_test + 256, test_number)

                acc_test = sess.run(accuracy, feed_dict={input_data: data_test_in[start_test:end_test],
                                                     input_labels: labels_test_in[start_test:end_test],
                                                     keep_prob: 1.0,
                                                     is_training: False})
                acc_list_test[i] = acc_test

            test_mean = np.mean(acc_list_test)
            fig_test[epoch] = test_mean
            max_test = np.max(fig_test)
            print("test_mean" + str(test_mean) + "------------maxtest" + str(max_test))
            if test_mean == max_test:
                saver = tf.train.Saver(var_list=var_list)
                saver_path = saver.save(sess, "Weights_Pavia/weights_pavia.ckpt")

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.plot(np.arange(iteration), fig_loss, label="Loss")

# 按一定间隔显示实现方法
# ax2.plot(200 * np.arange(len(fig_accuracy)), fig_accuracy, 'r')
lns2 = ax2.plot(np.arange(iteration), fig_accuracy, 'r', label="Accuracy")
ax1.set_xlabel('iteration')
ax1.set_ylabel('training loss')
ax2.set_ylabel('training accuracy')
# 合并图例
lns = lns1 + lns2
labels = ["Loss", "Accuracy"]
# labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=7)
plt.show()
#
fig2, ax3 = plt.subplots()
lns3 = ax3.plot(np.arange(iteration), fig_test, label="test_acc")
ax3.set_xlabel('iteration')
ax3.set_ylabel('test_acc')
lns4 = lns3
labels_5 = ["acc"]
# labels = [l.get_label() for l in lns]
plt.legend(lns4, labels_5, loc=7)
plt.show()