#!/usr/bin/env python
# coding=utf-8

import glob
import logging
import shutil
import datetime
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("job_name", 'fdn_train', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("num_threads", 64, "Number of threads")
tf.app.flags.DEFINE_integer("embedding_size", 16, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 1024, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 100, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_float("l1_reg", 0.0001, "L1 regularization")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {SGD, Adam, Adagrad, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '../../data/aliccp', "data dir")
tf.app.flags.DEFINE_string("model_dir", './model/fdn', "code check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", './model/fdn',
                           "export servable code for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing code or not")
tf.app.flags.DEFINE_string("pos_weights", "200,3000", "positive sample weight")
tf.app.flags.DEFINE_integer("task_num", 2, "task num")
tf.app.flags.DEFINE_string("vocab_index", '../../data/aliccp/vocab/', "feature index table")

tf.app.flags.DEFINE_string("dcp_nums", '2,2', 'dcp_nums')
tf.app.flags.DEFINE_string("task_specific_expert_units", '128,64', 'task_specific_expert_units')
tf.app.flags.DEFINE_string("shared_encoder_units", '128,64', 'shared_encoder_units')
tf.app.flags.DEFINE_string("loss_weights", '1.0,1.0', 'loss_weights')

# log level
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
current_time = datetime.datetime.now()
f_handler = logging.FileHandler(f'{current_time}_info_log.log')
formatter = logging.Formatter("%(levelname)s - %(asctime)s: %(message)s")
ch.setFormatter(formatter)
f_handler.setFormatter(formatter)
f_handler.setLevel(logging.INFO)

logger.addHandler(ch)
logger.addHandler(f_handler)

feat_101_index_file_path = FLAGS.vocab_index + 'vocab_101'
feat_109_14_index_file_path = FLAGS.vocab_index + 'vocab_109_14'
feat_110_14_index_file_path = FLAGS.vocab_index + 'vocab_110_14'
feat_127_14_index_file_path = FLAGS.vocab_index + 'vocab_127_14'
feat_150_14_index_file_path = FLAGS.vocab_index + 'vocab_150_14'
feat_121_index_file_path = FLAGS.vocab_index + 'vocab_121'
feat_122_index_file_path = FLAGS.vocab_index + 'vocab_122'
feat_124_index_file_path = FLAGS.vocab_index + 'vocab_124'
feat_125_index_file_path = FLAGS.vocab_index + 'vocab_125'
feat_126_index_file_path = FLAGS.vocab_index + 'vocab_126'
feat_127_index_file_path = FLAGS.vocab_index + 'vocab_127'
feat_128_index_file_path = FLAGS.vocab_index + 'vocab_128'
feat_129_index_file_path = FLAGS.vocab_index + 'vocab_129'
feat_205_index_file_path = FLAGS.vocab_index + 'vocab_205'
feat_206_index_file_path = FLAGS.vocab_index + 'vocab_206'
feat_207_index_file_path = FLAGS.vocab_index + 'vocab_207'
feat_210_index_file_path = FLAGS.vocab_index + 'vocab_210'
feat_216_index_file_path = FLAGS.vocab_index + 'vocab_216'
feat_508_index_file_path = FLAGS.vocab_index + 'vocab_508'
feat_509_index_file_path = FLAGS.vocab_index + 'vocab_509'
feat_702_index_file_path = FLAGS.vocab_index + 'vocab_702'
feat_853_index_file_path = FLAGS.vocab_index + 'vocab_853'
feat_301_index_file_path = FLAGS.vocab_index + 'vocab_301'

vocabs_list = [feat_101_index_file_path,feat_109_14_index_file_path,feat_110_14_index_file_path,feat_127_14_index_file_path,feat_150_14_index_file_path,
              feat_121_index_file_path, feat_122_index_file_path,feat_124_index_file_path,feat_125_index_file_path,feat_126_index_file_path,feat_127_index_file_path,
               feat_128_index_file_path,feat_129_index_file_path, feat_205_index_file_path,feat_206_index_file_path,feat_207_index_file_path,feat_210_index_file_path,
                feat_216_index_file_path,feat_508_index_file_path,feat_509_index_file_path,feat_702_index_file_path,feat_853_index_file_path,feat_301_index_file_path
               ]


def load_index_len(vocabs_list):
    vocab_dict = {}
    for vfile in vocabs_list:
        slot = vfile.split('/')[-1].split('_',1)[1]
        feat_key = f'feat_{slot}'
        with open(vfile,'r') as f:
            lines = f.readlines()
            vocab_dict[feat_key] = len(lines)+1
    return vocab_dict

vocab_dict = load_index_len(vocabs_list)
print("[INFO] vocab_dict:",vocab_dict)

def decode_line(line):
    """
    parse line data
    :param line:
    :return:
    """
    columns = tf.string_split([line], sep=',', skip_empty=False)
    # label解析 CTR  CVR
    labels = tf.string_to_number(columns.values[1: 3], out_type=tf.float32)

    # 所有特征域
    # 101  109_14 110_14 127_14 150_14 121 122 124 125 126 127 128 129 205 206 207 210 216 508 509 702 853 301
    # 特征解析
    feat_101 = tf.string_split([columns.values[3]], sep=':', skip_empty=False).values[1]
    feat_109_14 = tf.string_split([columns.values[4]], sep=':', skip_empty=False).values[1]
    feat_110_14 = tf.string_split([columns.values[5]], sep=':', skip_empty=False).values[1]
    feat_127_14 = tf.string_split([columns.values[6]], sep=':', skip_empty=False).values[1]
    feat_150_14 = tf.string_split([columns.values[7]], sep=':', skip_empty=False).values[1]
    feat_121 = tf.string_split([columns.values[8]], sep=':', skip_empty=False).values[1]
    feat_122 = tf.string_split([columns.values[9]], sep=':', skip_empty=False).values[1]
    feat_124 = tf.string_split([columns.values[10]], sep=':', skip_empty=False).values[1]
    feat_125 = tf.string_split([columns.values[11]], sep=':', skip_empty=False).values[1]
    feat_126 = tf.string_split([columns.values[12]], sep=':', skip_empty=False).values[1]
    feat_127 = tf.string_split([columns.values[13]], sep=':', skip_empty=False).values[1]
    feat_128 = tf.string_split([columns.values[14]], sep=':', skip_empty=False).values[1]
    feat_129 = tf.string_split([columns.values[15]], sep=':', skip_empty=False).values[1]
    feat_205 = tf.string_split([columns.values[16]], sep=':', skip_empty=False).values[1]
    feat_206 = tf.string_split([columns.values[17]], sep=':', skip_empty=False).values[1]
    feat_207 = tf.string_split([columns.values[18]], sep=':', skip_empty=False).values[1]
    feat_210 = tf.string_split([columns.values[19]], sep=':', skip_empty=False).values[1]
    feat_216 = tf.string_split([columns.values[20]], sep=':', skip_empty=False).values[1]
    feat_508 = tf.string_split([columns.values[21]], sep=':', skip_empty=False).values[1]
    feat_509 = tf.string_split([columns.values[22]], sep=':', skip_empty=False).values[1]
    feat_702 = tf.string_split([columns.values[23]], sep=':', skip_empty=False).values[1]
    feat_853 = tf.string_split([columns.values[24]], sep=':', skip_empty=False).values[1]
    feat_301 = tf.string_split([columns.values[25]], sep=':', skip_empty=False).values[1]

    return {
               'feat_101': feat_101,
               'feat_109_14': feat_109_14,
               'feat_110_14': feat_110_14,
               'feat_127_14': feat_127_14,
               'feat_150_14': feat_150_14,
               'feat_121': feat_121,
               'feat_122': feat_122,
               'feat_124': feat_124,
               'feat_125': feat_125,
               'feat_126': feat_126,
               'feat_127': feat_127,
               'feat_128': feat_128,
               'feat_129': feat_129,
               'feat_205': feat_205,
               'feat_206': feat_206,
               'feat_207': feat_207,
               'feat_210': feat_210,
               'feat_216': feat_216,
               'feat_508': feat_508,
               'feat_509': feat_509,
               'feat_702': feat_702,
               'feat_853': feat_853,
               'feat_301': feat_301
           }, labels




def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    files = tf.data.Dataset.list_files(filenames)

    # 多线程解析libsvm数据
    dataset = files.apply(
        tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.TextLineDataset(filename, buffer_size=batch_size * 32,  # compression_type='GZIP',
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


def variable_length_feature_process(var_len_feat, vocab_file, emb_params, sep='#',
                                    default_value=-1, sp_weights=None, combiner='sum'):
    """
    variable length feature processing
    :param var_len_feat: string type feature
    :param vocab_file: vocabulary files
    :param sep: separator
    :param default_value:
    :param sp_weights:
    :param combiner:
    :return:
    """
    feat_splited = tf.string_split(var_len_feat, sep=sep, skip_empty=False)
    feat_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocab_file, num_oov_buckets=1, default_value=default_value
    )
    sp_ids = tf.SparseTensor(
        indices=feat_splited.indices,
        values=feat_table.lookup(feat_splited.values),
        dense_shape=feat_splited.dense_shape)
    # emb_params = tf.Variable(tf.truncated_normal([vocab_len + 1, FLAGS.embedding_size]))
    return tf.nn.embedding_lookup_sparse(emb_params, sp_ids=sp_ids, sp_weights=sp_weights, combiner=combiner)


def model_fn(features, labels, mode, params):
    """build Estimator model"""

    common_wgts = []
    feat_101 = features['feat_101']
    feat_101_vocab_len = vocab_dict['feat_101']
    feat_101_wgts = tf.get_variable(name='feat_101_wgts',
                                    shape=[feat_101_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_101_wgts)
    feat_101_emb = variable_length_feature_process(feat_101, vocab_file=feat_101_index_file_path,
                                                   emb_params=feat_101_wgts)  # None * E
    feat_101_emb = tf.reshape(feat_101_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_109_14 = features['feat_109_14']
    feat_109_14_vocab_len = vocab_dict['feat_109_14']
    feat_109_14_wgts = tf.get_variable(name='feat_109_14_wgts',
                                       shape=[feat_109_14_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                       regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_109_14_wgts)
    feat_109_14_emb = variable_length_feature_process(feat_109_14, vocab_file=feat_109_14_index_file_path,
                                                      emb_params=feat_109_14_wgts)  # None * E
    feat_109_14_emb = tf.reshape(feat_109_14_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_110_14 = features['feat_110_14']
    feat_110_14_vocab_len = vocab_dict['feat_110_14']
    feat_110_14_wgts = tf.get_variable(name='feat_110_14_wgts',
                                       shape=[feat_110_14_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                       regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_110_14_wgts)
    feat_110_14_emb = variable_length_feature_process(feat_110_14, vocab_file=feat_110_14_index_file_path,
                                                      emb_params=feat_110_14_wgts)  # None * E
    feat_110_14_emb = tf.reshape(feat_110_14_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_127_14 = features['feat_127_14']
    feat_127_14_vocab_len = vocab_dict['feat_127_14']
    feat_127_14_wgts = tf.get_variable(name='feat_127_14_wgts',
                                       shape=[feat_127_14_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                       regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_127_14_wgts)
    feat_127_14_emb = variable_length_feature_process(feat_127_14, vocab_file=feat_127_14_index_file_path,
                                                      emb_params=feat_127_14_wgts)  # None * E
    feat_127_14_emb = tf.reshape(feat_127_14_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_150_14 = features['feat_150_14']
    feat_150_14_vocab_len = vocab_dict['feat_150_14']
    feat_150_14_wgts = tf.get_variable(name='feat_150_14_wgts',
                                       shape=[feat_150_14_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                       initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                       regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_150_14_wgts)
    feat_150_14_emb = variable_length_feature_process(feat_150_14, vocab_file=feat_150_14_index_file_path,
                                                      emb_params=feat_150_14_wgts)  # None * E
    feat_150_14_emb = tf.reshape(feat_150_14_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_121 = features['feat_121']
    feat_121_vocab_len = vocab_dict['feat_121']
    feat_121_wgts = tf.get_variable(name='feat_121_wgts',
                                    shape=[feat_121_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_121_wgts)
    feat_121_emb = variable_length_feature_process(feat_121, vocab_file=feat_121_index_file_path,
                                                   emb_params=feat_121_wgts)  # None * E
    feat_121_emb = tf.reshape(feat_121_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_122 = features['feat_122']
    feat_122_vocab_len = vocab_dict['feat_122']
    feat_122_wgts = tf.get_variable(name='feat_122_wgts',
                                    shape=[feat_122_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_122_wgts)
    feat_122_emb = variable_length_feature_process(feat_122, vocab_file=feat_122_index_file_path,
                                                   emb_params=feat_122_wgts)  # None * E
    feat_122_emb = tf.reshape(feat_122_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_124 = features['feat_124']
    feat_124_vocab_len = vocab_dict['feat_124']
    feat_124_wgts = tf.get_variable(name='feat_124_wgts',
                                    shape=[feat_124_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_124_wgts)
    feat_124_emb = variable_length_feature_process(feat_124, vocab_file=feat_124_index_file_path,
                                                   emb_params=feat_124_wgts)  # None * E
    feat_124_emb = tf.reshape(feat_124_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_125 = features['feat_125']
    feat_125_vocab_len = vocab_dict['feat_125']
    feat_125_wgts = tf.get_variable(name='feat_125_wgts',
                                    shape=[feat_125_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_125_wgts)
    feat_125_emb = variable_length_feature_process(feat_125, vocab_file=feat_125_index_file_path,
                                                   emb_params=feat_125_wgts)  # None * E
    feat_125_emb = tf.reshape(feat_125_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_126 = features['feat_126']
    feat_126_vocab_len = vocab_dict['feat_126']
    feat_126_wgts = tf.get_variable(name='feat_126_wgts',
                                    shape=[feat_126_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_126_wgts)
    feat_126_emb = variable_length_feature_process(feat_126, vocab_file=feat_126_index_file_path,
                                                   emb_params=feat_126_wgts)  # None * E
    feat_126_emb = tf.reshape(feat_126_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_127 = features['feat_127']
    feat_127_vocab_len = vocab_dict['feat_127']
    feat_127_wgts = tf.get_variable(name='feat_127_wgts',
                                    shape=[feat_127_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_127_wgts)
    feat_127_emb = variable_length_feature_process(feat_127, vocab_file=feat_127_index_file_path,
                                                   emb_params=feat_127_wgts)  # None * E
    feat_127_emb = tf.reshape(feat_127_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_128 = features['feat_128']
    feat_128_vocab_len = vocab_dict['feat_128']
    feat_128_wgts = tf.get_variable(name='feat_128_wgts',
                                    shape=[feat_128_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_128_wgts)
    feat_128_emb = variable_length_feature_process(feat_128, vocab_file=feat_128_index_file_path,
                                                   emb_params=feat_128_wgts)  # None * E
    feat_128_emb = tf.reshape(feat_128_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_129 = features['feat_129']
    feat_129_vocab_len = vocab_dict['feat_129']
    feat_129_wgts = tf.get_variable(name='feat_129_wgts',
                                    shape=[feat_129_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_129_wgts)
    feat_129_emb = variable_length_feature_process(feat_129, vocab_file=feat_129_index_file_path,
                                                   emb_params=feat_129_wgts)  # None * E
    feat_129_emb = tf.reshape(feat_129_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_205 = features['feat_205']
    feat_205_vocab_len =vocab_dict['feat_205']
    feat_205_wgts = tf.get_variable(name='feat_205_wgts',
                                    shape=[feat_205_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_205_wgts)
    feat_205_emb = variable_length_feature_process(feat_205, vocab_file=feat_205_index_file_path,
                                                   emb_params=feat_205_wgts)  # None * E
    feat_205_emb = tf.reshape(feat_205_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_206 = features['feat_206']
    feat_206_vocab_len = vocab_dict['feat_206']
    feat_206_wgts = tf.get_variable(name='feat_206_wgts',
                                    shape=[feat_206_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_206_wgts)
    feat_206_emb = variable_length_feature_process(feat_206, vocab_file=feat_206_index_file_path,
                                                   emb_params=feat_206_wgts)  # None * E
    feat_206_emb = tf.reshape(feat_206_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_207 = features['feat_207']
    feat_207_vocab_len = vocab_dict['feat_207']
    feat_207_wgts = tf.get_variable(name='feat_207_wgts',
                                    shape=[feat_207_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_207_wgts)
    feat_207_emb = variable_length_feature_process(feat_207, vocab_file=feat_207_index_file_path,
                                                   emb_params=feat_207_wgts)  # None * E
    feat_207_emb = tf.reshape(feat_207_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_210 = features['feat_210']
    feat_210_vocab_len = vocab_dict['feat_210']
    feat_210_wgts = tf.get_variable(name='feat_210_wgts',
                                    shape=[feat_210_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_210_wgts)
    feat_210_emb = variable_length_feature_process(feat_210, vocab_file=feat_210_index_file_path,
                                                   emb_params=feat_210_wgts)  # None * E
    feat_210_emb = tf.reshape(feat_210_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_216 = features['feat_216']
    feat_216_vocab_len = vocab_dict['feat_216']
    feat_216_wgts = tf.get_variable(name='feat_216_wgts',
                                    shape=[feat_216_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_216_wgts)
    feat_216_emb = variable_length_feature_process(feat_216, vocab_file=feat_216_index_file_path,
                                                   emb_params=feat_216_wgts)  # None * E
    feat_216_emb = tf.reshape(feat_216_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_508 = features['feat_508']
    feat_508_vocab_len = vocab_dict['feat_508']
    feat_508_wgts = tf.get_variable(name='feat_508_wgts',
                                    shape=[feat_508_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_508_wgts)
    feat_508_emb = variable_length_feature_process(feat_508, vocab_file=feat_508_index_file_path,
                                                   emb_params=feat_508_wgts)  # None * E
    feat_508_emb = tf.reshape(feat_508_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_509 = features['feat_509']
    feat_509_vocab_len = vocab_dict['feat_509']
    feat_509_wgts = tf.get_variable(name='feat_509_wgts',
                                    shape=[feat_509_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_509_wgts)
    feat_509_emb = variable_length_feature_process(feat_509, vocab_file=feat_509_index_file_path,
                                                   emb_params=feat_509_wgts)  # None * E
    feat_509_emb = tf.reshape(feat_509_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_702 = features['feat_702']
    feat_702_vocab_len = vocab_dict['feat_702']
    feat_702_wgts = tf.get_variable(name='feat_702_wgts',
                                    shape=[feat_702_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_702_wgts)
    feat_702_emb = variable_length_feature_process(feat_702, vocab_file=feat_702_index_file_path,
                                                   emb_params=feat_702_wgts)  # None * E
    feat_702_emb = tf.reshape(feat_702_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_853 = features['feat_853']
    feat_853_vocab_len = vocab_dict['feat_853']
    feat_853_wgts = tf.get_variable(name='feat_853_wgts',
                                    shape=[feat_853_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_853_wgts)
    feat_853_emb = variable_length_feature_process(feat_853, vocab_file=feat_853_index_file_path,
                                                   emb_params=feat_853_wgts)  # None * E
    feat_853_emb = tf.reshape(feat_853_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    feat_301 = features['feat_301']
    feat_301_vocab_len = vocab_dict['feat_301']
    feat_301_wgts = tf.get_variable(name='feat_301_wgts',
                                    shape=[feat_301_vocab_len, FLAGS.embedding_size], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=(2 / 512) ** 0.5),
                                    regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg))
    common_wgts.append(feat_301_wgts)
    feat_301_emb = variable_length_feature_process(feat_301, vocab_file=feat_301_index_file_path,
                                                   emb_params=feat_301_wgts)  # None * E
    feat_301_emb = tf.reshape(feat_301_emb, shape=[-1, 1, FLAGS.embedding_size])  # None * 1 * E

    embedding = tf.concat(
        [feat_101_emb, feat_109_14_emb, feat_110_14_emb, feat_127_14_emb, feat_150_14_emb, feat_121_emb, feat_122_emb,
         feat_124_emb, feat_125_emb, feat_126_emb, feat_127_emb, feat_128_emb, feat_129_emb, feat_205_emb, feat_206_emb,
         feat_207_emb, feat_210_emb, feat_216_emb, feat_508_emb, feat_509_emb, feat_702_emb, feat_853_emb,
         feat_301_emb], axis=-1)  # None * 1 * (23 * E)

    embedding = tf.layers.batch_normalization(embedding)
    embedding = tf.reshape(embedding, [-1, 23 * FLAGS.embedding_size])  # None * (F * E)

    # task-specific output of each task represents the current task and is used to constrain the private encoder of the task
    task_specific_outputs = []
    # Shared input, characterizing shared features
    task_shared_outputs = []
    with tf.variable_scope("DCP-part"):
        dcp_nums = list(map(int, FLAGS.dcp_nums.strip().split(',')))
        for i in range(FLAGS.task_num):
            task_specific_output = []
            for j in range(dcp_nums[i]):
                # DCP combination
                # task-specific experts
                task_specific_expert_units = list(map(int, FLAGS.task_specific_expert_units.strip().split(',')))
                task_specific_expert = embedding
                l1_reg = tf.contrib.layers.l1_regularizer(FLAGS.l1_reg)
                for k in range(len(task_specific_expert_units)):
                    task_specific_expert = tf.contrib.layers.fully_connected(inputs=task_specific_expert,
                                                                        num_outputs=task_specific_expert_units[k],
                                                                        activation_fn=tf.nn.elu,
                                                                        weights_regularizer=l1_reg,
                                                                        scope='private_encoder_%d_%d_%d' % (
                                                                            i, j, k))  # None * 64
                specific_shape = task_specific_expert.get_shape().as_list()
                task_specific_output.append(
                    tf.reshape(task_specific_expert, shape=[-1, 1, specific_shape[-1]]))  # None * 1 * 64
                # task-shared experts
                task_shared_expert_units = list(map(int, FLAGS.shared_encoder_units.strip().split(',')))
                task_shared_expert = embedding
                for k in range(len(task_shared_expert_units)):
                    task_shared_expert = tf.contrib.layers.fully_connected(inputs=task_shared_expert,
                                                                       num_outputs=task_shared_expert_units[k],
                                                                       activation_fn=tf.nn.elu,
                                                                       scope='shared_encoder_%d_%d_%d' % (
                                                                           i, j, k))  # None * 64
                shared_shape = task_specific_expert.get_shape().as_list()
                task_shared_outputs.append(tf.reshape(task_shared_expert, shape=[-1, 1, shared_shape[-1]]))
            task_specific_outputs.append(task_specific_output)

    # feature fusion constraint
    shared_feat = tf.concat(task_shared_outputs, axis=1)  # None * N * F

    task_gate_outputs = []
    # task main output
    for i in range(FLAGS.task_num):
        task_out = tf.concat(task_specific_outputs[i] + [shared_feat], axis=1)
        task_out_shape = task_out.get_shape().as_list()
        task_gate_outputs.append(tf.reshape(task_out, shape=[-1, task_out_shape[1] * task_out_shape[2]]))

    # ctr
    y_ctr = tf.concat(task_gate_outputs[0], axis=-1)
    y_ctr = tf.contrib.layers.fully_connected(inputs=y_ctr, num_outputs=1,
                                              activation_fn=None, \
                                              weights_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg),
                                              scope='deep_out_ctr')
    y_ctr = tf.reshape(y_ctr, [-1, ])
    y_ctr_prediction = tf.sigmoid(y_ctr)

    # cvr
    y_cvr = tf.concat(task_gate_outputs[1], axis=-1)
    y_cvr = tf.contrib.layers.fully_connected(inputs=y_cvr, num_outputs=1,
                                              activation_fn=None, \
                                              weights_regularizer=tf.contrib.layers.l2_regularizer(FLAGS.l2_reg),
                                              scope='deep_out_cvr')
    y_cvr = tf.reshape(y_cvr, [-1, ])
    y_cvr_prediction = tf.sigmoid(y_cvr)

    if mode != tf.estimator.ModeKeys.PREDICT:
        # ------label split------
        labels = tf.split(labels, num_or_size_splits=2, axis=-1)
        label_ctr = tf.reshape(labels[0], shape=[-1, ])
        label_cvr = tf.reshape(labels[1], shape=[-1, ])
        targets = []
        targets.append(label_ctr)
        targets.append(label_cvr)

    predictions = {
        "prob": 0.995 * y_ctr_prediction + 0.005 * y_cvr_prediction,
        "click": y_ctr,
        "valid_play": y_cvr
    }
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}
    # Estimator predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    # ------label split and build loss function------
    loss_weights = list(map(float, FLAGS.loss_weights.strip().split(',')))
    with tf.variable_scope("loss-function-part"):
        loss = loss_weights[0] * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=y_ctr, targets=label_ctr,
                                                     pos_weight=float(FLAGS.pos_weights.split(',')[0]))) + \
               loss_weights[1] * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=y_cvr, targets=label_cvr,
                                                     pos_weight=float(FLAGS.pos_weights.split(',')[1])))

        # auxiliary task constraint
        y_sigmoids = []
        for i in range(FLAGS.task_num):
            y_sigmoid = []
            for out in task_specific_outputs[i]:
                y_ = tf.contrib.layers.fully_connected(inputs=tf.reshape(out, shape=[-1, specific_shape[-1]]),
                                                       num_outputs=1,
                                                       activation_fn=None,
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                           FLAGS.l2_reg)
                                                       )
                y_ = tf.reshape(y_, [-1, ])
                y_sigmoid.append(tf.sigmoid(y_))
                loss += loss_weights[i] * tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(logits=y_, targets=targets[i],
                                                             pos_weight=float(FLAGS.pos_weights.split(',')[i])))
                # loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, labels=targets[i]))
            y_sigmoids.append(y_sigmoid)

        # orthogonal constraint processing
        def difference_loss(private_encoder, shared_encoder):
            """
            Frobenius norm
            :param private_encoder:
            :param shared_encoder:
            :return:
            """
            shared_encoder = tf.transpose(shared_encoder, perm=[0, 2, 1])  # None * 64 * 1
            matmul = tf.matmul(shared_encoder, private_encoder) + 1e-6  # None * 64 * 64
            cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(matmul), axis=-1), axis=-1)))
            return cost

        for task_specific_expert, task_shared_expert in zip([private_encoder for private_enncoders in task_specific_outputs
                                                    for private_encoder in private_enncoders], task_shared_outputs):
            loss += difference_loss(task_specific_expert, task_shared_expert)

    # Provide an estimator spec for `ModeKeys.EVAL`
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "auc": tf.metrics.auc(label_ctr, y_ctr_prediction),
            "auc_ctr": tf.metrics.auc(label_ctr, y_ctr_prediction),
            "auc_cvr": tf.metrics.auc(label_cvr, y_cvr_prediction),
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate, initial_accumulator_value=1e-6)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)

def main(_):
    # ------init Envs------
    print(FLAGS.data_dir)
    tr_files = glob.glob("%s/train_data/train_data.csv" % FLAGS.data_dir)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/train_data/train_data.csv" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/train_data/train_data.csv" % FLAGS.data_dir)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing code cleaned at %s" % FLAGS.model_dir)

    # strategy = tf.distribute.MirroredStrategy(
        # cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())  # train_distribute=strategy, eval_distribute=strategy
    # config_proto = tf.ConfigProto(allow_soft_placement=True,
    #                               device_count={'CPU': 1},
    #                               intra_op_parallelism_threads=0,
    #                               inter_op_parallelism_threads=0,
    #                               log_device_placement=False
    #                               )
    config_proto = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 1})
    config = tf.estimator.RunConfig(session_config=config_proto,
                                    log_step_count_steps=FLAGS.log_steps, save_checkpoints_steps=FLAGS.log_steps,
                                    save_summary_steps=FLAGS.log_steps, tf_random_seed=2021)
             
    # config = tf.estimator.RunConfig(train_distribute=strategy, eval_distribute=strategy, session_config=config_proto,
    #                                 log_step_count_steps=FLAGS.log_steps, save_checkpoints_steps=FLAGS.log_steps * 80,
    #                                 save_summary_steps=FLAGS.log_steps * 80, tf_random_seed=2021)

    Model = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, config=config)

    feature_spec = {
        'feat_101': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_101'),
        'feat_109_14': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_109_14'),
        'feat_110_14': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_110_14'),
        'feat_127_14': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_127_14'),
        'feat_150_14': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_150_14'),
        'feat_121': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_121'),
        'feat_122': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_122'),
        'feat_124': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_124'),
        'feat_125': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_125'),
        'feat_126': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_126'),
        'feat_127': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_127'),
        'feat_128': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_128'),
        'feat_129': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_129'),
        'feat_205': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_205'),
        'feat_206': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_206'),
        'feat_207': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_207'),
        'feat_210': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_210'),
        'feat_216': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_216'),
        'feat_508': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_508'),
        'feat_509': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_509'),
        'feat_702': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_702'),
        'feat_853': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_853'),
        'feat_301': tf.placeholder(dtype=tf.string, shape=[None, ], name='feat_301')
    }

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs,
                                      batch_size=FLAGS.batch_size))

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size),
            steps=None,
            start_delay_secs=1200, throttle_secs=1200
        )
        tf.estimator.train_and_evaluate(Model, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        Model.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'export':
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        Model.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
