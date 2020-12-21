import tensorflow as tf

weights = {"weights_oct1_hh": tf.get_variable(name="weights_oct1_hh", shape=[5, 3, 3, 1, 24],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_oct1_hl": tf.get_variable(name="weights_oct1_hl", shape=[5, 3, 3, 1, 24],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_oct1_high": tf.get_variable(name='biases_oct1_high', shape=[24],
                                               initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_oct1_low": tf.get_variable(name='biases_oct1_low', shape=[24],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),

           "weights_oct2_hh": tf.get_variable(name="weights_oct2_hh", shape=[5, 3, 3, 24, 48],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_oct2_hl": tf.get_variable(name="weights_oct2_hl", shape=[5, 3, 3, 24, 48],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_oct2_lh": tf.get_variable(name="weights_oct2_lh", shape=[5, 3, 3, 24, 48],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_oct2_ll": tf.get_variable(name="weights_oct2_ll", shape=[5, 3, 3, 24, 48],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_oct2_high": tf.get_variable(name='biases_oct2_high', shape=[48],
                                               initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_oct2_low": tf.get_variable(name='biases_oct2_low', shape=[48],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),

           "weights_oct3_hh": tf.get_variable(name="weights_oct3_hh", shape=[5, 3, 3, 48, 24],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_oct3_hl": tf.get_variable(name="weights_oct3_hl", shape=[5, 3, 3, 48, 24],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_oct3_high": tf.get_variable(name='biases_oct3_high', shape=[24],
                                               initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_oct3_low": tf.get_variable(name='biases_oct3_low', shape=[24],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),

           "weights_oct4_hh": tf.get_variable(name="weights_oct4_hh", shape=[5, 3, 3, 24, 1],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_oct4_hl": tf.get_variable(name="weights_oct4_hl", shape=[5, 3, 3, 24, 1],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_oct4_lh": tf.get_variable(name="weights_oct4_lh", shape=[5, 3, 3, 24, 1],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_oct4_ll": tf.get_variable(name="weights_oct4_ll", shape=[5, 3, 3, 24, 1],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_oct4_high": tf.get_variable(name='biases_oct4_high', shape=[1],
                                               initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_oct4_low": tf.get_variable(name='biases_oct4_low', shape=[1],
                                              initializer=tf.random_normal_initializer(stddev=0.01)),

           "weights_convA": tf.get_variable(name="weights_convA", shape=[3, 3, 103, 103],
                                            initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_convA": tf.get_variable(name='biases_convA', shape=[103],
                                           initializer=tf.random_normal_initializer(stddev=0.01)),

           "weights_spatial_pixel": tf.get_variable(name="weights_spatial_pixel", shape=[1, 1, 103, 103],
                                                    initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_spatial_pixel": tf.get_variable(name="biases_spatial_pixel", shape=[103],
                                                   initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_spectral_pixel": tf.get_variable(name="weights_spectral_pixel", shape=[1, 1, 103, 103],
                                                     initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_spectral_pixel": tf.get_variable(name="biases_spectral_pixel", shape=[103],
                                                    initializer=tf.random_normal_initializer(stddev=0.01)),

           "weights_spatial_fc1": tf.get_variable(name='weights_spatial_fc1', shape=[7 * 7 * 103, 1024],
                                                  initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_spatial_fc1": tf.get_variable(name='biases_spatial_fc1', shape=[1024],
                                                 initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_spatial_fc2": tf.get_variable(name='weights_spatial_fc2', shape=[1024, 9],
                                                  initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_spatial_fc2": tf.get_variable(name='biases_spatial_fc2', shape=[9],
                                                 initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_spectral_fc1": tf.get_variable(name='weights_spectral_fc1', shape=[7 * 7 * 103, 1024],
                                                   initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_spectral_fc1": tf.get_variable(name='biases_spectral_fc1', shape=[1024],
                                                  initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_spectral_fc2": tf.get_variable(name='weights_spectral_fc2', shape=[1024, 9],
                                                   initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_spectral_fc2": tf.get_variable(name='biases_spectral_fc2', shape=[9],
                                                  initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_fc1": tf.get_variable(name='weights_fc1', shape=[7 * 7 * 103, 1024],
                                          initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_fc1": tf.get_variable(name='biases_fc1', shape=[1024],
                                         initializer=tf.random_normal_initializer(stddev=0.01)),
           "weights_fc2": tf.get_variable(name='weights_fc2', shape=[1024, 9],
                                          initializer=tf.random_normal_initializer(stddev=0.01)),
           "biases_fc2": tf.get_variable(name='biases_fc2', shape=[9],
                                         initializer=tf.random_normal_initializer(stddev=0.01))}


def batch_normal(x, is_training):
    data_bn = tf.contrib.slim.batch_norm(inputs=x, is_training=is_training, updates_collections=None)
    return data_bn


def octconv_1(weights, input_high):
    data_dim = tf.reshape(input_high, [-1, 13, 13, 103, 1])
    aa = tf.transpose(data_dim, [0, 3, 1, 2, 4])

    data_hh = tf.nn.conv3d(aa, weights["weights_oct1_hh"], strides=[1, 1, 1, 1, 1], padding="SAME")
    high_to_low = tf.nn.max_pool3d(aa, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding="SAME")
    data_hl = tf.nn.conv3d(high_to_low, weights["weights_oct1_hl"], strides=[1, 1, 1, 1, 1], padding="SAME")
    return data_hh, data_hl


def octconv_2(weights, input_high, input_low):
    data_hh = tf.nn.conv3d(input_high, weights["weights_oct2_hh"], strides=[1, 1, 1, 1, 1], padding="SAME")
    data_down = tf.nn.max_pool3d(input_high, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding="SAME")
    data_hl = tf.nn.conv3d(data_down, weights["weights_oct2_hl"], strides=[1, 1, 1, 1, 1], padding="SAME")

    data_lh_conv = tf.nn.conv3d(input_low, weights["weights_oct2_lh"], strides=[1, 1, 1, 1, 1], padding="SAME")
    aa = tf.transpose(data_lh_conv, [0, 2, 3, 1, 4])
    bb = tf.reshape(aa, [-1, 7, 7, 103 * 48])
    data_lh = tf.image.resize_images(images=bb, size=[13, 13], method=tf.image.ResizeMethod.BILINEAR)
    cc = tf.reshape(data_lh, [-1, 13, 13, 103, 48])
    dd = tf.transpose(cc, [0, 3, 1, 2, 4])

    data_ll = tf.nn.conv3d(input_low, weights["weights_oct2_ll"], strides=[1, 1, 1, 1, 1], padding="SAME")
    data_high = data_hh + dd
    data_low = data_hl + data_ll
    return data_high, data_low


def octconv_3(weights, input_high):
    data_hh = tf.nn.conv3d(input_high, weights["weights_oct3_hh"], strides=[1, 1, 1, 1, 1], padding="SAME")
    high_to_low = tf.nn.avg_pool3d(input_high, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding="SAME")
    data_hl = tf.nn.conv3d(high_to_low, weights["weights_oct3_hl"], strides=[1, 1, 1, 1, 1], padding="SAME")
    return data_hh, data_hl


def octconv_4(weights, input_high, input_low):
    data_hh = tf.nn.conv3d(input_high, weights["weights_oct4_hh"], strides=[1, 1, 1, 1, 1], padding="SAME")
    data_down = tf.nn.avg_pool3d(input_high, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding="SAME")
    data_hl = tf.nn.conv3d(data_down, weights["weights_oct4_hl"], strides=[1, 1, 1, 1, 1], padding="SAME")

    data_lh_conv = tf.nn.conv3d(input_low, weights["weights_oct4_lh"], strides=[1, 1, 1, 1, 1], padding="SAME")
    aa = tf.transpose(data_lh_conv, [0, 2, 3, 1, 4])
    bb = tf.reshape(aa, [-1, 4, 4, 103])
    data_lh = tf.image.resize_images(images=bb, size=[7, 7], method=tf.image.ResizeMethod.BILINEAR)
    cc = tf.reshape(data_lh, [-1, 7, 7, 103, 1])
    dd = tf.transpose(cc, [0, 3, 1, 2, 4])

    data_ll = tf.nn.conv3d(input_low, weights["weights_oct4_ll"], strides=[1, 1, 1, 1, 1], padding="SAME")
    data_high = data_hh + dd
    data_low = data_hl + data_ll
    return data_high, data_low


def res_block(input, weights, is_training):
    octconv1_high, octconv1_low = octconv_1(weights, input)
    biases1_high = tf.nn.bias_add(octconv1_high, weights["biases_oct1_high"])
    biases1_low = tf.nn.bias_add(octconv1_low, weights["biases_oct1_low"])
    bn1_high = batch_normal(biases1_high, is_training)
    bn1_low = batch_normal(biases1_low, is_training)
    relu1_high = tf.nn.relu(bn1_high)
    relu1_low = tf.nn.relu(bn1_low)

    octconv2_high, octconv2_low = octconv_2(weights, relu1_high, relu1_low)
    biases2_high = tf.nn.bias_add(octconv2_high, weights["biases_oct2_high"])
    biases2_low = tf.nn.bias_add(octconv2_low, weights["biases_oct2_low"])
    bn2_high = batch_normal(biases2_high, is_training)
    bn2_low = batch_normal(biases2_low, is_training)
    relu2_high = tf.nn.relu(bn2_high)
    relu2_low = tf.nn.relu(bn2_low)

    pool_high = tf.nn.max_pool3d(relu2_high, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding="SAME")
    data_fusion1 = pool_high + relu2_low

    octconv3_high, octconv3_low = octconv_3(weights, data_fusion1)
    biases3_high = tf.nn.bias_add(octconv3_high, weights["biases_oct3_high"])
    biases3_low = tf.nn.bias_add(octconv3_low, weights["biases_oct3_low"])
    bn3_high = batch_normal(biases3_high, is_training)
    bn3_low = batch_normal(biases3_low, is_training)
    relu3_high = tf.nn.relu(bn3_high)
    relu3_low = tf.nn.relu(bn3_low)

    octconv4_high, octconv4_low = octconv_4(weights, relu3_high, relu3_low)
    biases4_high = tf.nn.bias_add(octconv4_high, weights["biases_oct4_high"])
    biases4_low = tf.nn.bias_add(octconv4_low, weights["biases_oct4_low"])
    bn4_high = batch_normal(biases4_high, is_training)
    bn4_low = batch_normal(biases4_low, is_training)
    relu4_high = tf.nn.relu(bn4_high)
    relu4_low = tf.nn.relu(bn4_low)

    aa = tf.transpose(relu4_high, [0, 2, 3, 1, 4])
    bb = tf.reshape(aa, [-1, 7, 7, 103])

    cc = tf.transpose(relu4_low, [0, 2, 3, 1, 4])
    dd = tf.reshape(cc, [-1, 4, 4, 103])

    relu4_low_up = tf.image.resize_images(images=dd, size=[7, 7], method=tf.image.ResizeMethod.BILINEAR)
    out = bb + relu4_low_up
    return out


def spatial_attention(input_spatial, weights, is_training):
    spatial_convA = tf.nn.conv2d(input_spatial, weights["weights_convA"], strides=[1, 1, 1, 1], padding="SAME")
    spatial_biasesA = tf.nn.bias_add(spatial_convA, weights["biases_convA"])
    spatial_bnA = batch_normal(spatial_biasesA, is_training)
    spatial_reluA = tf.nn.relu(spatial_bnA)

    spatialA_reshape_1 = tf.reshape(spatial_reluA, [-1, 7 * 7, 103])
    spatialA_transpose = tf.transpose(spatialA_reshape_1, [0, 2, 1])
    spatialA_matmul_1 = tf.matmul(spatialA_reshape_1, spatialA_transpose)
    spatialA_Softmax = tf.nn.softmax(spatialA_matmul_1)
    spatialA_matmul_2 = tf.matmul(spatialA_Softmax, spatialA_reshape_1)
    spatialA_reshape_2 = tf.reshape(spatialA_matmul_2, [-1, 7, 7, 103])
    spatial_feature = input_spatial + spatialA_reshape_2

    spatial_conv = tf.nn.conv2d(spatial_feature, weights["weights_spatial_pixel"], strides=[1, 1, 1, 1],
                                padding="VALID")
    spatial_biases = tf.nn.bias_add(spatial_conv, weights["biases_spatial_pixel"])
    spatial_bn = batch_normal(spatial_biases, is_training)
    spatial_relu = tf.nn.relu(spatial_bn)
    return spatial_relu


def spectral_attention(input_spectral, weights, is_training):
    spectral_reshape_1 = tf.reshape(input_spectral, [-1, 7 * 7, 103])
    spectral_transpose = tf.transpose(spectral_reshape_1, [0, 2, 1])
    spectral_matmul_1 = tf.matmul(spectral_transpose, spectral_reshape_1)
    spectral_softmax = tf.nn.softmax(spectral_matmul_1)
    spectral_matmul_2 = tf.matmul(spectral_reshape_1, spectral_softmax)
    spectral_reshape_2 = tf.reshape(spectral_matmul_2, [-1, 7, 7, 103])
    spectral_feature = input_spectral + spectral_reshape_2

    spectral_conv = tf.nn.conv2d(spectral_feature, weights["weights_spectral_pixel"], strides=[1, 1, 1, 1],
                                 padding="VALID")
    spectral_biases = tf.nn.bias_add(spectral_conv, weights["biases_spectral_pixel"])
    spectral_bn = batch_normal(spectral_biases, is_training)
    spectral_relu = tf.nn.relu(spectral_bn)
    return spectral_relu


def feature_fusion(input_spatial, input_spectral):
    spatial_reshape = tf.reshape(input_spatial, [-1, 7 * 7, 103])
    spectral_reshape = tf.reshape(input_spectral, [-1, 7 * 7, 103])
    spatial_transpose = tf.transpose(spatial_reshape, [0, 2, 1])
    spectral_transpose = tf.transpose(spectral_reshape, [0, 2, 1])
    spe_to_spa = tf.matmul(spatial_reshape, spectral_transpose)
    spa_to_spe = tf.matmul(spectral_reshape, spatial_transpose)
    spatial_soft = tf.nn.softmax(spe_to_spa)
    spectral_soft = tf.nn.softmax(spa_to_spe)
    spatial_final = tf.matmul(spatial_soft, spectral_reshape) + spatial_reshape
    spectral_final = tf.matmul(spectral_soft, spatial_reshape) + spectral_reshape
    feature_sum = spatial_final + spectral_final
    return spatial_final, spectral_final, feature_sum


def logits_spatial(input_spatial, weights, keep_prob):
    spatial_reshape = tf.reshape(input_spatial, [-1, 7 * 7 * 103])
    spatial_logit1 = tf.matmul(spatial_reshape, weights["weights_spatial_fc1"]) + weights["biases_spatial_fc1"]
    spatial_fc1 = tf.nn.relu(spatial_logit1)
    spatial_drop = tf.nn.dropout(spatial_fc1, keep_prob=keep_prob)
    spatial_logits = tf.matmul(spatial_drop, weights["weights_spatial_fc2"]) + weights["biases_spatial_fc2"]
    return spatial_logits


def logits_spectral(input_spectral, weights, keep_prob):
    spectral_reshape = tf.reshape(input_spectral, [-1, 7 * 7 * 103])
    spectral_logit1 = tf.matmul(spectral_reshape, weights["weights_spectral_fc1"]) + weights["biases_spectral_fc1"]
    spectral_fc1 = tf.nn.relu(spectral_logit1)
    spectral_drop = tf.nn.dropout(spectral_fc1, keep_prob=keep_prob)
    spectral_logits = tf.matmul(spectral_drop, weights["weights_spectral_fc2"]) + weights["biases_spectral_fc2"]
    return spectral_logits


def classer(input_fusion, weights, keep_prob):
    fusion_reshape = tf.reshape(input_fusion, [-1, 7 * 7 * 103])
    fusion_logit1 = tf.matmul(fusion_reshape, weights["weights_fc1"]) + weights["biases_fc1"]
    fusion_fc1 = tf.nn.relu(fusion_logit1)
    fusion_drop = tf.nn.dropout(fusion_fc1, keep_prob=keep_prob)
    logits = tf.matmul(fusion_drop, weights["weights_fc2"]) + weights["biases_fc2"]
    prediction = tf.nn.softmax(logits)
    return logits, prediction


def network(input, weights, keep_prob, is_training):
    feature_res = res_block(input, weights, is_training)
    feature_spatial = spatial_attention(feature_res, weights, is_training)
    feature_spectral = spectral_attention(feature_res, weights, is_training)
    spa_sum, spe_sum, fusion = feature_fusion(feature_spatial, feature_spectral)
    logit_spatial = logits_spatial(spa_sum, weights, keep_prob)
    logit_spectral = logits_spectral(spe_sum, weights, keep_prob)
    logit, predict = classer(fusion, weights, keep_prob)
    return logit_spatial, logit_spectral, logit, predict