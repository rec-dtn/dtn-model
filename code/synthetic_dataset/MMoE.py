#!/usr/bin/env python
# coding=utf-8

import glob
import logging
import math
import random
import shutil

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.estimator.canned import metric_keys
import os as os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name", 'mmoe_train', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("num_threads", 64, "线程数，Number of threads")
tf.app.flags.DEFINE_integer("embedding_size", 32, "embedding大小，Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "训练轮次，Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 1024, "训练批次大小，Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 100, "保存频率，save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "学习率，正数，learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.01, "L2正则系数，L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "损失评估类型，loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "训练优化器，optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '128,64', "深度层设置，deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5', "dropout设置，与deep_layers设置对应，dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "是否进行批归一化，perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '../../data/synthetic_dataset/', "数据主目录，data dir")

tf.app.flags.DEFINE_string("model_dir", './model/mmoe', "cp存放，code check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", './model/mmoe',
                           "模型存放，export servable code for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "任务类型，task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", True, "是否清除已存在的模型，clear existing code or not")
tf.app.flags.DEFINE_float("pos_weight", 1, "正样本权重")
tf.app.flags.DEFINE_integer("experts_num", 8, "打分专家个数")
tf.app.flags.DEFINE_integer("task_num", 2, "任务数")
tf.app.flags.DEFINE_integer("units", 32, "打分专家神经元个数")

# 日志打印设置
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
ch = logging.StreamHandler()  # 标准输出流
formatter = logging.Formatter("%(levelname)s - %(asctime)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def decode_line(line):
    """
    解码数据
    :param line:
    :return:
    """
    columns = tf.string_split([line], sep=',', skip_empty=False)
    # label解析
    labels = tf.string_to_number(columns.values[0 : 2], out_type=tf.float32)

    # 特征解析
    features = tf.string_to_number(columns.values[1026: ], out_type=tf.float32)

    return {
               'features': features
           }, labels


def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    files = tf.data.Dataset.list_files(filenames)

    # 多线程解析libsvm数据
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.TextLineDataset(filename, buffer_size=batch_size * 32, #compression_type='GZIP',
                                                     num_parallel_reads=10),
            cycle_length=len(filenames),
            buffer_output_elements=batch_size,
            prefetch_input_elements=batch_size,
            sloppy=True))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda line:
                                                               decode_line(line),
                                                               batch_size=batch_size,
                                                               num_parallel_batches=10))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 32)
    dataset = dataset.repeat(num_epochs)
    dataset.prefetch(400000)
    dataset.cache()

    return dataset


def valid_input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    files = tf.data.Dataset.list_files(filenames)

    # 多线程解析libsvm数据
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.TextLineDataset(filename, buffer_size=batch_size * 32,
                                                     num_parallel_reads=10),
            cycle_length=len(filenames),
            buffer_output_elements=batch_size,
            prefetch_input_elements=batch_size,
            sloppy=True))
    if perform_shuffle:
        num_parallel_batches = 10
        buffer_size = 200000
    else:
        num_parallel_batches = 50
        buffer_size = 1000000
    dataset = dataset.apply(tf.data.experimental.map_and_batch(lambda line:
                                                               decode_line(line),
                                                               batch_size=batch_size,
                                                               num_parallel_batches=num_parallel_batches))
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 4)
    dataset = dataset.repeat(num_epochs)
    dataset.prefetch(500000)
    dataset.cache()
    return dataset


def model_fn(features, labels, mode, params):
    print("开启新一轮训练~")
    """构建Estimator模型"""
    # ------hyperparameters----
    dropout_ = FLAGS.dropout
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    # 多目标优化参数设置
    experts_num = FLAGS.experts_num
    task_num = FLAGS.task_num
    units = FLAGS.units

    # ------获取特征输入-------
    features_ = features['features']
    # numerical_feat = tf.log1p(features_) # None * 499
    feat_dim = 512
    embedding = tf.reshape(features_, shape=[-1, feat_dim])
    # numerical_feat = tf.reshape(numerical_feat, shape=[-1, 499, 1])
    # numerical_w = tf.get_variable(name='numerical_w',
    #                               shape=[499, FLAGS.embedding_size], dtype=tf.float32,
    #                               initializer=tf.glorot_normal_initializer())
    # numerical_feat = tf.multiply(numerical_feat, numerical_w)  # None * 499 * E
    #
    # embedding = tf.concat([numerical_feat], axis=1)  # None * 499 * E

    # 我的理解实现方式
    experts = []  # [None * 64...]
    with tf.variable_scope("experts-part"):
        for j in range(experts_num):
            y_dnn = tf.reshape(embedding, [-1, feat_dim])  # None * (F * E)
            # 128,64
            dnn_layer_nodes = list(map(int, FLAGS.deep_layers.split(',')))
            dropout = list(map(float, dropout_.split(',')))
            for i in range(len(dnn_layer_nodes)):
                y_dnn = tf.contrib.layers.fully_connected(inputs=y_dnn, num_outputs=dnn_layer_nodes[i],
                                                          activation_fn=tf.nn.relu, \
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                          scope='dnn_%d_%d' % (j, i))
                if mode == tf.estimator.ModeKeys.TRAIN:
                    y_dnn = tf.nn.dropout(y_dnn, keep_prob=dropout[i])
            experts.append(tf.expand_dims(y_dnn, axis=1))  # None * 1 * 64

    experts = tf.concat(experts, axis=1)  # None * 8 * 64

    final_outputs = []

    gate_networks = [tf.reshape(tf.contrib.layers.fully_connected(inputs=embedding, num_outputs=experts_num,
                                                          activation_fn=tf.nn.relu, \
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                          scope='gate_%d' % (i)), shape = [-1, experts_num, 1])
                     for i in range(task_num)]

    for gate_network in gate_networks:
        task_out = tf.multiply(experts, gate_network)
        final_outputs.append(tf.reduce_sum(task_out, axis = 1))

    def bulid_tower(x, first_dnn_size=128, second_dnn_size=64, activation_fn=tf.nn.relu):
        y_tower = tf.contrib.layers.fully_connected(inputs=x, num_outputs=first_dnn_size, activation_fn=activation_fn, \
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), )
        y_tower = tf.contrib.layers.fully_connected(inputs=y_tower, num_outputs=second_dnn_size,
                                                    activation_fn=activation_fn, \
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), )
        return y_tower

    # task1
    y_1 = tf.concat(final_outputs[0], axis=-1)
    y_1 = bulid_tower(y_1)
    y_1 = tf.contrib.layers.fully_connected(inputs=y_1, num_outputs=1, activation_fn=None, \
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                scope='task1')
    y_1 = tf.reshape(y_1, [-1, ])

    # task2
    y_2 = tf.concat(final_outputs[1], axis=-1)
    y_2 = bulid_tower(y_2)
    y_2 = tf.contrib.layers.fully_connected(inputs=y_2, num_outputs=1, activation_fn=None, \
                                               weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                               scope='task2_out')
    y_2 = tf.reshape(y_2, [-1, ])


    # 预测结果导出格式设置
    predictions = {
        "y_1": y_1,
        "y_2": y_2
    }
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}
    # Estimator预测模式
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------拆分标签，构建损失------
    labels = tf.split(labels, num_or_size_splits=2, axis=-1)
    label_1 = tf.reshape(labels[0], shape=[-1, ])
    label_2 = tf.reshape(labels[1], shape=[-1, ])
    loss = tf.reduce_mean(
        tf.losses.mean_squared_error(predictions=y_1, labels=label_1)) + \
           tf.reduce_mean(
        tf.losses.mean_squared_error(predictions=y_2, labels=label_2)) #+ \
           # l2_reg * tf.nn.l2_loss(numerical_w)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "mse": tf.metrics.mean_absolute_error(label_1, y_1),
        "mse_1": tf.metrics.mean_absolute_error(label_1, y_1),
        "mse_2": tf.metrics.mean_absolute_error(label_2, y_2)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-6)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-6)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def batch_norm_layer(x, train_phase, scope_bn):
    """
    批标准化
    :param x:
    :param train_phase:
    :param scope_bn:
    :return:
    """
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def main(_):
    # ------init Envs------
    print(FLAGS.data_dir)
    tr_files = glob.glob('%s/train_data/*' % FLAGS.data_dir)
    random.shuffle([tr_files])
    print("tr_files:", tr_files)
    va_files = glob.glob('%s/test_data/*' % FLAGS.data_dir)
    print("va_files:", va_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing code cleaned at %s" % FLAGS.model_dir)

    # ------bulid Tasks------
    model_params = {
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
    }

    # strategy = tf.contrib.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1","/gpu:2","/gpu:3"])  # train_distribute=strategy, eval_distribute=strategy
    strategy = tf.distribute.MirroredStrategy()  # train_distribute=strategy, eval_distribute=strategy
    # strategy = tf.distribute.experimental.CentralStorageStrategy(
    # )  # train_distribute=strategy, eval_distribute=strategy
    config_proto = tf.ConfigProto(allow_soft_placement=True,
                                  device_count={'GPU': 4},
                                  intra_op_parallelism_threads=0,
                                  # 线程池中线程的数量，一些独立的操作可以在这指定的数量的线程中进行并行，如果设置为0代表让系统设置合适的数值
                                  inter_op_parallelism_threads=0,
                                  # 每个进程可用的为进行阻塞操作节点准备的线程池中线程的数量，设置为0代表让系统选择合适的数值，负数表示所有的操作在调用者的线程中进行。注意：如果在创建第一个Session的适合制定了该选项，那么之后创建的所有Session都会保持一样的设置，除非use_per_session_threads为true或配置了session_inter_op_thread_pool。
                                  log_device_placement=False,
                                  # gpu_options=gpu_options
                                  )
    config = tf.estimator.RunConfig(train_distribute=strategy, eval_distribute=strategy, session_config=config_proto,
                                    log_step_count_steps=FLAGS.log_steps, save_checkpoints_steps=FLAGS.log_steps * 10,
                                    save_summary_steps=FLAGS.log_steps * 10, tf_random_seed=2021)

    MMoE = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    feature_spec = {
        'features': tf.placeholder(dtype=tf.float32, shape=[None, 512], name='features'),
    }

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs,
                                      batch_size=FLAGS.batch_size))

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size),
            steps=None,
            # exporters=[model_best_exporter(FLAGS.job_name, feature_spec, exports_to_keep=1,
            #                                metric_key=metric_keys.MetricKeys.AUC, big_better=False)],
            start_delay_secs=10, throttle_secs=10
        )
        tf.estimator.train_and_evaluate(MMoE, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        MMoE.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
