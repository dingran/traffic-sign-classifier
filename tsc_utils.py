""" This file contains all the utility functions that will be called by Traffic_Sign_Classifier.ipynb"""
import numpy as np
from tqdm import tqdm
import cv2
import tensorflow as tf
import time
import os
from sklearn.utils import shuffle
import inspect
import pickle
import matplotlib.pyplot as plt
import hashlib

DEFAULT_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def load_pickled_data(file, columns):
    """
    Loads pickled training and test data.

    Parameters
    ----------
    file    :
              Name of the pickle file.
    columns : list of strings
              List of columns in pickled data we're interested in.

    Returns
    -------
    A tuple of datasets for given columns.
    """

    with open(file, mode='rb') as f:
        dataset = pickle.load(f)
    return tuple(map(lambda c: dataset[c], columns))


def generate_data(X_train, y_train, X_valid, y_valid, X_test, y_test,
                  flip=True, augment=True, aug_factor=1.0,
                  distort=True, resize=True, rotate=True, shift=True, blursharpen=True,
                  boost=False, target_y=[26, 21],
                  use_grayscale=True, keep_original=False):
    if flip:
        X_train, y_train = flip_extend(X_train, y_train)
        X_valid, y_valid = flip_extend(X_valid, y_valid)

    if augment:
        setting = dict(distort=distort,
                       resize=resize,
                       rotate=rotate,
                       shift=shift,
                       blursharpen=blursharpen)
        boost = boost

        if boost:
            target_y = target_y
            N_copy = 10
            X_train_boost, y_train_boost = augment_data(X_train, y_train, N_copy=N_copy, target_y=target_y, **setting)
            X_valid_boost, y_valid_boost = augment_data(X_valid, y_valid, N_copy=N_copy, target_y=target_y, **setting)

        X_train, y_train = augment_data(X_train, y_train, **setting, factor=aug_factor)
        X_valid, y_valid = augment_data(X_valid, y_valid, **setting, factor=aug_factor)

        if boost:
            X_train = np.concatenate([X_train, X_train_boost], axis=0)
            y_train = np.concatenate([y_train, y_train_boost], axis=0)
            X_valid = np.concatenate([X_valid, X_valid_boost], axis=0)
            y_valid = np.concatenate([y_valid, y_valid_boost], axis=0)

    print(X_train.shape, y_train.shape)
    print(X_valid.shape, y_valid.shape)

    # color conversion, hist equalization and normalization
    X_train = preprocess_data(X_train, use_grayscale=use_grayscale, keep_original=keep_original)
    X_valid = preprocess_data(X_valid, use_grayscale=use_grayscale, keep_original=keep_original)
    X_test = preprocess_data(X_test, use_grayscale=use_grayscale, keep_original=keep_original)

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)


def preprocess_data(X, use_grayscale=True, keep_original=False, equalize_all=False, normed=True):
    assert X.shape[1] == X.shape[2] == 32

    print('Input shapes: ', X.shape)
    X_out = []
    clahe = DEFAULT_CLAHE

    if use_grayscale:
        color_method = cv2.COLOR_BGR2GRAY
        print('Using gray')
    else:
        color_method = cv2.COLOR_BGR2YUV
        print('Using YUV')

    for xx in tqdm(X):
        res = cv2.cvtColor(xx, color_method)
        if len(res.shape) < 3:
            res = np.expand_dims(res, axis=2)

        if keep_original:
            res = np.concatenate([res, xx], axis=2)

        if equalize_all:
            for i in range(res.shape[-1]):
                res[:, :, i] = clahe.apply(res[:, :, i])
        else:  # only sharpening channel 0, assuming this is a grayscale channel!!!!!
            res[:, :, 0] = clahe.apply(res[:, :, 0])

        if normed:
            res = res.astype(float)
            for i in range(res.shape[-1]):  # normalize to 0 mean, and 1 stdev
                res[:, :, i] = (res[:, :, i] - res[:, :, i].mean()) / res[:, :, i].std()

        X_out.append(res)

    X_out = np.asarray(X_out)
    print('Output shapes: ', X_out.shape)
    return X_out


def augment_data(X, y, distort=True, resize=True, rotate=True, shift=True, blursharpen=True,
                 N_copy=1, target_y=None, factor=1.0):
    print('========= augment_data() arguments: =========')
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args[2:]:
        print("{} = {}".format(i, values[i]))
    print('=============================================')

    print('Input shapes: ', X.shape, y.shape)
    X_out = []
    Y_out = []

    for xx, yy in zip(tqdm(X), y):

        if target_y is not None:
            if yy not in target_y:
                continue

        if target_y is None:
            # the original
            X_out.append(xx)
            Y_out.append(yy)

        for i in range(N_copy):
            if distort:
                for d in [3, 5]:
                    X_out.append(distort_img(xx, d_limit=d * factor))
                    Y_out.append(yy)

            if resize:
                for s in np.concatenate([[0.9, 1.1], np.random.uniform(0.8, 1.2, 2)]):
                    X_out.append(resize_img(xx, scale=s * factor))
                    Y_out.append(yy)
            if rotate:
                for r in np.concatenate([[-15, 15], np.random.uniform(-20, 20, 2)]):
                    X_out.append(rotate_img(xx, angle=r * factor))
                    Y_out.append(yy)

            if shift:
                for dxdy in np.random.uniform(-4, 4, (4, 2)):
                    X_out.append(shift_img(xx, dx=dxdy[0] * factor, dy=dxdy[1] * factor))
                    Y_out.append(yy)

            if blursharpen:
                b, s = blur_and_sharpen_img(xx, factor=factor)
                X_out.append(b)
                Y_out.append(yy)
                X_out.append(s)
                Y_out.append(yy)

    X_out = np.asarray(X_out)
    Y_out = np.asarray(Y_out)
    print('Output shapes: ', X_out.shape, Y_out.shape)
    return X_out, Y_out


params_orig_lenet = dict(conv1_k=5, conv1_d=6, conv1_p=0.95,
                         conv2_k=5, conv2_d=16, conv2_p=0.95,
                         fc3_size=120, fc3_p=0.5,
                         fc4_size=84, fc4_p=0.5,
                         num_classes=43, model_name='lenet', name='orig_lenet')

params_big_lenet = dict(conv1_k=5, conv1_d=6 * 4, conv1_p=0.8,
                        conv2_k=5, conv2_d=16 * 4, conv2_p=0.8,
                        fc3_size=120 * 4, fc3_p=0.5,
                        fc4_size=84 * 3, fc4_p=0.5,
                        num_classes=43, model_name='lenet', name='big_lenet')

params_huge_lenet = dict(conv1_k=5, conv1_d=6 * 8, conv1_p=0.8,
                         conv2_k=5, conv2_d=16 * 8, conv2_p=0.8,
                         fc3_size=120 * 8, fc3_p=0.5,
                         fc4_size=84 * 6, fc4_p=0.5,
                         num_classes=43, model_name='lenet', name='huge_lenet')


def lenet(x, params, is_training):
    print(params)
    do_batch_norm = False
    if 'batch_norm' in params.keys():
        if params['batch_norm']:
            do_batch_norm = True

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(x, kernel_size=params['conv1_k'], depth=params['conv1_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params['conv1_p']), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params['conv2_k'], depth=params['conv2_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params['conv2_p']), lambda: pool2)

    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])
    print('lenet pool2 reshaped size: ', pool2.get_shape().as_list())

    with tf.variable_scope('fc3'):
        fc3 = fully_connected_relu(pool2, size=params['fc3_size'], is_training=is_training, BN=do_batch_norm)
        fc3 = tf.cond(is_training, lambda: tf.nn.dropout(fc3, keep_prob=params['fc3_p']), lambda: fc3)

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(fc3, size=params['fc4_size'], is_training=is_training, BN=do_batch_norm)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params['fc4_p']), lambda: fc4)

    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size=params['num_classes'], is_training=is_training)

    return logits


params_sermanet_v2 = dict(conv1_k=5, conv1_d=32, conv1_p=0.9,
                          conv2_k=5, conv2_d=64, conv2_p=0.8,
                          conv3_k=5, conv3_d=128, conv3_p=0.7,
                          fc4_size=1024, fc4_p=0.5,
                          num_classes=43, model_name='sermanet_v2', name='standard')

params_sermanet_v2_big = dict(conv1_k=5, conv1_d=32 * 2, conv1_p=0.9,
                              conv2_k=5, conv2_d=64 * 2, conv2_p=0.8,
                              conv3_k=5, conv3_d=128 * 2, conv3_p=0.7,
                              fc4_size=1024 * 2, fc4_p=0.5,
                              num_classes=43, model_name='sermanet_v2', name='big')


def sermanet_v2(x, params, is_training):
    print(params)
    do_batch_norm = False
    if 'batch_norm' in params.keys():
        if params['batch_norm']:
            do_batch_norm = True

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(x, kernel_size=params['conv1_k'], depth=params['conv1_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params['conv1_p']), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params['conv2_k'], depth=params['conv2_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params['conv2_p']), lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size=params['conv3_k'], depth=params['conv3_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool3'):
        pool3 = pool(conv3, size=2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob=params['conv3_p']), lambda: pool3)

    # Fully connected

    # 1st stage output
    pool1 = pool(pool1, size=4)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])

    # 2nd stage output
    pool2 = pool(pool2, size=2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

    # 3rd stage output
    shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])

    flattened = tf.concat([pool1, pool2, pool3], 1)

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size=params['fc4_size'], is_training=is_training, BN=do_batch_norm)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params['fc4_p']), lambda: fc4)
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size=params['num_classes'], is_training=is_training)
    return logits


params_sermanet = dict(conv1_k=5, conv1_d=108, conv1_p=0.9,
                       conv2_k=5, conv2_d=108, conv2_p=0.8,
                       fc4_size=100, fc4_p=0.5,
                       num_classes=43, model_name='sermanet', name='standard')

params_sermanet_big = dict(conv1_k=5, conv1_d=100, conv1_p=0.9,
                           conv2_k=5, conv2_d=200, conv2_p=0.8,
                           fc4_size=200, fc4_p=0.5,
                           num_classes=43, model_name='sermanet', name='big')


def sermanet(x, params, is_training):
    print(params)
    do_batch_norm = False
    if 'batch_norm' in params.keys():
        if params['batch_norm']:
            do_batch_norm = True

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(x, kernel_size=params['conv1_k'], depth=params['conv1_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool1'):
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params['conv1_p']), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params['conv2_k'], depth=params['conv2_d'], is_training=is_training,
                          BN=do_batch_norm)
    with tf.variable_scope('pool2'):
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params['conv2_p']), lambda: pool2)

    # Fully connected

    # 1st stage output
    pool1 = pool(pool1, size=2)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])

    # 2nd stage output
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

    flattened = tf.concat([pool1, pool2], 1)

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size=params['fc4_size'], is_training=is_training, BN=do_batch_norm)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params['fc4_p']), lambda: fc4)
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size=params['num_classes'], is_training=is_training)
    return logits


def train_model(X_train, y_train, X_valid, y_valid, X_test, y_test,
                resuming=False,
                model=lenet, model_params=params_orig_lenet,
                learning_rate=0.001, max_epochs=1001, batch_size=256,
                early_stopping_enabled=True, early_stopping_patience=10,
                log_epoch=1, print_epoch=1,
                top_k=5, return_top_k=False,
                plot_featuremap=False):
    print('========= train_model() arguments: ==========')
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args[6:]:
        print("{} = {}".format(i, values[i]))
    print('=============================================')

    fn = ''
    model_name = model_params.pop('name', None)
    print(model_name)
    for k in sorted(model_params.keys()):
        if k != 'num_classes' and k != 'model_name' and k != 'batch_norm':
            fn += k + '_' + str(model_params[k]) + '_'
    data_str = ''
    if X_test.shape[-1] > 1:
        data_str = str(X_test.shape[-1])
    model_id = model_params['model_name'] + data_str + '__' + fn[:-1]

    if 'batch_norm' in model_params.keys():
        if model_params['batch_norm']:
            model_id = 'BN_' + model_id

    model_id_hash = str(hashlib.sha1(model_id.encode('utf-8')).hexdigest()[-16:])
    print(model_id)
    print(model_id_hash)
    model_dir = os.path.join(os.getcwd(), 'models', model_id_hash)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, model_id), exist_ok=True)
    print('model dir: {}'.format(model_dir))
    model_fname = os.path.join(model_dir, 'model_cpkt')
    model_fname_best_epoch = os.path.join(model_dir, 'best_epoch')
    model_train_history = os.path.join(model_dir, 'training_history.npz')

    start = time.time()

    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
        x = tf.placeholder(tf.float32, (None, 32, 32, X_test.shape[-1]))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, model_params['num_classes'])
        is_training = tf.placeholder(tf.bool)

        logits = model(x, params=model_params, is_training=is_training)

        predictions = tf.nn.softmax(logits)
        top_k_predictions = tf.nn.top_k(predictions, top_k)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
        loss_operation = tf.reduce_mean(cross_entropy, name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_operation = optimizer.minimize(loss_operation)
        pred_y = tf.argmax(logits, 1, name='prediction')
        actual_y = tf.argmax(one_hot_y, 1)
        correct_prediction = tf.equal(pred_y, actual_y)
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(variable)
            print(shape)
            # print(len(shape))
            variable_parametes = 1
            for dim in shape:
                # print(dim)
                variable_parametes *= dim.value
            # print(variable_parametes)
            total_parameters += variable_parametes
        print('total # of parameters: ', total_parameters)

        def output_top_k(X_data):
            top_k_preds = sess.run([top_k_predictions], feed_dict={x: X_data, is_training: False})
            return top_k_preds

        def evaluate(X_data, y_data, aux_output=False):
            n_data = len(X_data)
            correct_pred = np.array([])
            y_pred = np.array([])
            y_actual = np.array([])
            loss_batch = np.array([])
            acc_batch = np.array([])
            batch_sizes = np.array([])
            for offset in range(0, n_data, batch_size):
                batch_x, batch_y = X_data[offset:offset + batch_size], y_data[offset:offset + batch_size]
                batch_sizes = np.append(batch_sizes, batch_y.shape[0])

                if aux_output:
                    accuracy, loss, cp_, yp_, ya_ = \
                        sess.run([accuracy_operation, loss_operation, correct_prediction, pred_y, actual_y],
                                 feed_dict={x: batch_x, y: batch_y, is_training: False})

                    correct_pred = np.append(correct_pred, cp_)
                    y_pred = np.append(y_pred, yp_)
                    y_actual = np.append(y_actual, ya_)
                else:
                    accuracy, loss = sess.run([accuracy_operation, loss_operation],
                                              feed_dict={x: batch_x, y: batch_y, is_training: False})

                loss_batch = np.append(loss_batch, loss)
                acc_batch = np.append(acc_batch, accuracy)

            final_acc = np.average(acc_batch, weights=batch_sizes)
            final_loss = np.average(loss_batch, weights=batch_sizes)

            if aux_output:
                return final_acc, final_loss, correct_pred, y_pred, y_actual
            else:
                return final_acc, final_loss

        # If we chose to keep training previously trained model, restore session.
        if resuming:
            try:
                tf.train.Saver().restore(sess, model_fname)
                print('Restored session from {}'.format(model_fname))
            except Exception as e:
                print("Failed restoring previously trained model: file does not exist.")
                print("Trying to restore from best epoch from previously training session.")
                try:
                    tf.train.Saver().restore(sess, model_fname_best_epoch)
                    print('Restored session from {}'.format(model_fname_best_epoch))
                except Exception as e:
                    print("Failed to restore, will train from scratch now.")

                    # print([v.op.name for v in tf.all_variables()])
                    # print([n.name for n in tf.get_default_graph().as_graph_def().node])

        saver = tf.train.Saver()
        early_stopping = EarlyStopping(tf.train.Saver(), sess, patience=early_stopping_patience, minimize=True,
                                       restore_path=model_fname_best_epoch)

        train_loss_history = np.empty([0], dtype=np.float32)
        train_accuracy_history = np.empty([0], dtype=np.float32)
        valid_loss_history = np.empty([0], dtype=np.float32)
        valid_accuracy_history = np.empty([0], dtype=np.float32)
        if max_epochs > 0:
            print("================= TRAINING ==================")
        else:
            print("================== TESTING ==================")
        print(" Timestamp: " + get_time_hhmmss())

        for epoch in range(max_epochs):
            X_train, y_train = shuffle(X_train, y_train)

            for offset in tqdm(range(0, X_train.shape[0], batch_size)):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_training: True})

            # If another significant epoch ended, we log our losses.
            if epoch % log_epoch == 0:
                train_accuracy, train_loss = evaluate(X_train, y_train)
                valid_accuracy, valid_loss = evaluate(X_valid, y_valid)

                if epoch % print_epoch == 0:
                    print("-------------- EPOCH %4d/%d --------------" % (epoch, max_epochs))
                    print("     Train loss: %.8f, accuracy: %.2f%%" % (train_loss, 100 * train_accuracy))
                    print("Validation loss: %.8f, accuracy: %.2f%%" % (valid_loss, 100 * valid_accuracy))
                    print("      Best loss: %.8f at epoch %d" % (
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch))
                    print("   Elapsed time: " + get_time_hhmmss(start))
                    print("      Timestamp: " + get_time_hhmmss())
            else:
                valid_loss = 0.
                valid_accuracy = 0.
                train_loss = 0.
                train_accuracy = 0.

            valid_loss_history = np.append(valid_loss_history, [valid_loss])
            valid_accuracy_history = np.append(valid_accuracy_history, [valid_accuracy])
            train_loss_history = np.append(train_loss_history, [train_loss])
            train_accuracy_history = np.append(train_accuracy_history, [train_accuracy])

            if early_stopping_enabled:
                # Get validation data predictions and log validation loss:
                if valid_loss == 0:
                    _, valid_loss = evaluate(X_valid, y_valid)
                if early_stopping(valid_loss, epoch):
                    print("Early stopping.\nBest monitored loss was {:.8f} at epoch {}.".format(
                        early_stopping.best_monitored_value, early_stopping.best_monitored_epoch
                    ))
                    break

        # Evaluate on test dataset.
        valid_accuracy, valid_loss, valid_cp, valid_yp, valid_ya = evaluate(X_valid, y_valid, aux_output=True)
        test_accuracy, test_loss, test_cp, test_yp, test_ya = evaluate(X_test, y_test, aux_output=True)
        print("=============================================")
        print(" Valid loss: %.8f, accuracy = %.2f%%)" % (valid_loss, 100 * valid_accuracy))
        print(" Test loss: %.8f, accuracy = %.2f%%)" % (test_loss, 100 * test_accuracy))
        print(" Total time: " + get_time_hhmmss(start))
        print("  Timestamp: " + get_time_hhmmss())

        # Save model weights for future use.
        saved_model_path = saver.save(sess, model_fname)
        print("Model file: " + saved_model_path)
        np.savez(model_train_history, train_loss_history=train_loss_history,
                 train_accuracy_history=train_accuracy_history, valid_loss_history=valid_loss_history,
                 valid_accuracy_history=valid_accuracy_history)
        print("Train history file: " + model_train_history)

        if return_top_k:
            top_k_preds = output_top_k(X_test)

        def outputFeatureMap(image_input, tf_activation, title='', activation_min=-1, activation_max=-1, plt_num=1):
            # Here make sure to preprocess your image_input in a way your network expects
            # with size, normalization, ect if needed
            # image_input =
            # Note: x should be the same name as your network's tensorflow data placeholder variable
            # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
            activation = tf_activation.eval(session=sess, feed_dict={x: image_input, is_training: False})
            featuremaps = activation.shape[3]
            plt.figure(figsize=((featuremaps // 6 + 1) * 2, 6 * 2))

            for featuremap in range(featuremaps):
                plt.subplot(6, featuremaps // 6 + 1,
                            featuremap + 1)  # sets the number of feature maps to show on each row and column
                # plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
                if activation_min != -1 & activation_max != -1:
                    plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                               vmax=activation_max, cmap="gray")
                elif activation_max != -1:
                    plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max,
                               cmap="gray")
                elif activation_min != -1:
                    plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                               cmap="gray")
                else:
                    plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")

                plt.yticks([])
                plt.xticks([])

            plt.suptitle(title)
            plt.show()

        if plot_featuremap:
            for n_img in range(10):
                outputFeatureMap(np.expand_dims(X_test[n_img], axis=0),
                                 graph.get_operation_by_name('conv1/add').outputs[0],
                                 title='conv1/add, img: {}'.format(n_img))
                # outputFeatureMap(np.expand_dims(X_test[n_img], axis=0), graph.get_operation_by_name('conv1/Relu').outputs[0], title='conv1/Relu')

                # outputFeatureMap(np.expand_dims(X_test[n_img], axis=0), graph.get_operation_by_name('conv2/add').outputs[0], title='conv2/add')
                # outputFeatureMap(np.expand_dims(X_test[n_img], axis=0), graph.get_operation_by_name('conv2/Relu').outputs[0], title='conv2/Relu')

                # outputFeatureMap(np.expand_dims(X_test[n_img], axis=0), graph.get_operation_by_name('conv3/add').outputs[0], title='conv3/add')

    result_dict = dict(test_accuracy=test_accuracy, test_loss=test_loss, test_cp=test_cp, test_yp=test_yp,
                       test_ya=test_ya,
                       valid_accuracy=valid_accuracy, valid_loss=valid_loss, valid_cp=valid_cp, valid_yp=valid_yp,
                       valid_ya=valid_ya)
    if return_top_k:
        return result_dict, top_k_preds
    else:
        return result_dict


class EarlyStopping(object):
    """
    Provides early stopping functionality. Keeps track of model accuracy,
    and if it doesn't improve over time restores last best performing
    parameters.
    """

    def __init__(self, saver, session, patience=100, minimize=True, restore_path=None):
        """
        Initialises a `EarlyStopping` isntance.

        Parameters
        ----------
        saver     :
                    TensorFlow Saver object to be used for saving and restoring model.
        session   :
                    TensorFlow Session object containing graph where model is restored.
        patience  :
                    Early stopping patience. This is the number of epochs we wait for
                    accuracy to start improving again before stopping and restoring
                    previous best performing parameters.

        Returns
        -------
        New instance.
        """
        self.minimize = minimize
        self.patience = patience
        self.saver = saver
        self.session = session
        self.best_monitored_value = np.inf if minimize else 0.
        self.best_monitored_epoch = 0
        self.restore_path = restore_path

    def __call__(self, value, epoch):
        """
        Checks if we need to stop and restores the last well performing values if we do.

        Parameters
        ----------
        value     :
                    Last epoch monitored value.
        epoch     :
                    Last epoch number.

        Returns
        -------
        `True` if we waited enough and it's time to stop and we restored the
        best performing weights, or `False` otherwise.
        """
        if (self.minimize and value < self.best_monitored_value) or (
                    not self.minimize and value > self.best_monitored_value):
            self.best_monitored_value = value
            self.best_monitored_epoch = epoch
            self.saver.save(self.session, self.restore_path)
        elif self.best_monitored_epoch + self.patience < epoch:
            if self.restore_path is not None:
                self.saver.restore(self.session, self.restore_path)
            else:
                print("ERROR: Failed to restore session")
            return True

        return False


def fully_connected(input_x, size, is_training, BN=False):
    """
    Performs a single fully connected layer pass, e.g. returns `input * weights + bias`.
    """
    weights = tf.get_variable('weights',
                              shape=[input_x.get_shape()[1], size],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[size],
                             initializer=tf.constant_initializer(0.0)
                             )
    out = tf.matmul(input_x, weights) + biases
    if BN:
        out = tf.contrib.layers.batch_norm(out, center=True, scale=True, is_training=is_training, scope='bn')
    return out


def fully_connected_relu(input_x, size, is_training, BN=False):
    return tf.nn.relu(fully_connected(input_x, size, is_training, BN=BN))


def conv_relu(input_x, kernel_size, depth, is_training, BN=False):
    """
    Performs a single convolution layer pass.
    """
    weights = tf.get_variable('weights',
                              shape=[kernel_size, kernel_size, input_x.get_shape()[3], depth],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[depth],
                             initializer=tf.constant_initializer(0.0)
                             )
    conv = tf.nn.conv2d(input_x, weights, strides=[1, 1, 1, 1], padding='SAME')
    if BN:
        conv = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=is_training, scope='bn')
    return tf.nn.relu(conv + biases)


def pool(input_x, size):
    """
    Performs a max pooling layer pass.
    """
    return tf.nn.max_pool(input_x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def distort_img(input_img, d_limit=4):
    """
    Apply warpPerspective transformation on image, with 4 key points, randomly generated around the corners
    with uniform distribution with a range of [-d_limit, d_limit]
    :param input_img:
    :param d_limit:
    :return:
    """
    if d_limit == 0:
        return input_img
    rows, cols, ch = input_img.shape
    pts2 = np.float32([[0, 0], [rows - 1, 0], [0, cols - 1], [rows - 1, cols - 1]])
    pts1 = np.float32(pts2 + np.random.uniform(-d_limit, d_limit, pts2.shape))
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(input_img, M, (cols, rows), borderMode=1)
    return dst


def resize_img(input_img, scale=1.1):
    """
    Function to scale image content while keeping the overall image size, padding is done with border replication
    Scale > 1 means making content bigger
    :param input_img: X * Y * ch
    :param scale: positive real number
    :return: scaled image
    """
    if scale == 1.0:
        return input_img
    rows, cols, ch = input_img.shape
    d = rows * (scale - 1)  # overall image size change from rows, cols, to rows - 2d, cols - 2d
    pts1 = np.float32([[d, d], [rows - 1 - d, d], [d, cols - 1 - d], [rows - 1 - d, cols - 1 - d]])
    pts2 = np.float32([[0, 0], [rows - 1, 0], [0, cols - 1], [rows - 1, cols - 1]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(input_img, M, (cols, rows), borderMode=1)
    return dst


def rotate_img(input_img, angle=15):
    if angle == 0:
        return input_img
    rows, cols, ch = input_img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(input_img, M, (cols, rows), borderMode=1)
    return dst


def shift_img(input_img, dx=2, dy=2):
    if dx == 0 and dy == 0:
        return input_img
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(input_img, M, (input_img.shape[0], input_img.shape[1]), borderMode=1)
    return dst


def blur_and_sharpen_img(input_img, kernel=(3, 3), ratio=0.7, factor=1.0):
    blur = cv2.GaussianBlur(input_img, kernel, 0)
    sharp = cv2.addWeighted(input_img, 1.0 + ratio * factor, blur, -ratio * factor, 0)
    return blur, sharp


def flip_extend(X, y):
    """
    Credit: https://github.com/navoshta/traffic-signs/blob/master/Traffic_Signs_Recognition.ipynb

    Extends existing images dataset by flipping images of some classes. As some images would still belong
    to same class after flipping we extend such classes with flipped images. Images of other would toggle
    between two classes when flipped, so for those we extend existing datasets as well.

    Parameters
    ----------
    X       : ndarray
              Dataset array containing feature examples.
    y       : ndarray, optional, defaults to `None`
              Dataset labels in index form.

    Returns
    -------
    A tuple of X and y.
    """

    print('Input sizes: ', X.shape, y.shape)

    # Classes of signs that, when flipped horizontally, should still be classified as the same class
    self_flippable_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
    # Classes of signs that, when flipped vertically, should still be classified as the same class
    self_flippable_vertically = np.array([1, 5, 12, 15, 17])
    # Classes of signs that, when flipped horizontally and then vertically, should still be classified as the same class
    self_flippable_both = np.array([32, 40])
    # Classes of signs that, when flipped horizontally, would still be meaningful, but should be classified as some other class
    cross_flippable = np.array([
        [19, 20],
        [33, 34],
        [36, 37],
        [38, 39],
        [20, 19],
        [34, 33],
        [37, 36],
        [39, 38],
    ])
    num_classes = 43

    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    y_extended = np.empty([0], dtype=y.dtype)

    for c in tqdm(range(num_classes)):
        # First copy existing data for this class
        X_extended = np.append(X_extended, X[y == c], axis=0)
        # If we can flip images of this class horizontally and they would still belong to said class...
        if c in self_flippable_horizontally:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X[y == c][:, :, ::-1, :], axis=0)
        # If we can flip images of this class horizontally and they would belong to other class...
        if c in cross_flippable[:, 0]:
            # ...Copy flipped images of that other class to the extended array.
            flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(X_extended, X[y == flip_class][:, :, ::-1, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        # If we can flip images of this class vertically and they would still belong to said class...
        if c in self_flippable_vertically:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, :, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        # If we can flip images of this class horizontally AND vertically and they would still belong to said class...
        if c in self_flippable_both:
            # ...Copy their flipped versions into extended array.
            X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis=0)
        # Fill labels for added images set to current class.
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

    print('Output sizes: ', X_extended.shape, y_extended.shape)
    return X_extended, y_extended


def get_time_hhmmss(start=None):
    """
    Calculates time since `start` and formats as a string.
    """
    if start is None:
        return time.strftime("%Y/%m/%d %H:%M:%S")
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str
