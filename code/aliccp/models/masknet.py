"""
论文：MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask

地址：https://arxiv.org/pdf/2102.0761

代码：https://github.com/QunBB/DeepLearning/blob/main/recommendation/rank/masknet.py
"""
import tensorflow as tf
from typing import List, Union, Optional
from functools import partial
import importlib
from typing import List, Callable, Optional, Union


def dnn_layer(inputs: tf.Tensor,
              hidden_units: Union[List[int], int],
              activation: Optional[Union[Callable, str]] = None,
              dropout: Optional[float] = 0.,
              is_training: Optional[bool] = True,
              use_bn: Optional[bool] = True,
              l2_reg: float = 0.,
              use_bias: bool = True,
              scope=None):
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]

    output = inputs
    for idx, size in enumerate(hidden_units):
        output = tf.layers.dense(output, size,
                                 use_bias=use_bias,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                 kernel_initializer=tf.glorot_normal_initializer(),
                                 name=scope+f'_{idx}' if scope else None)
        if use_bn:
            output = tf.layers.batch_normalization(output, training=is_training, name=scope+f'_bn_{idx}' if scope else None)

        if activation is not None:
            output = activation_layer(activation, is_training=is_training, scope=f'activation_layer_{idx}')(output)

        if is_training:
            output = tf.nn.dropout(output, 1 - dropout)

    return output


nn_module = None


def activation_layer(activation: Union[Callable, str],
                     scope: Optional[str] = None,
                     is_training: bool = True):
    if isinstance(activation, str):
        if activation.lower() == 'dice':
            return lambda x: dice(x, is_training, scope if scope else '')
        elif activation.lower() == 'prelu':
            return lambda x: prelu(x, scope if scope else '')
        else:
            global nn_module
            if nn_module is None:
                nn_module = importlib.import_module('tensorflow.nn')
                return getattr(nn_module, activation)
    else:
        if activation is dice:
            return lambda x: dice(x, is_training, scope if scope else '')
        elif activation is prelu:
            return lambda x: prelu(x, scope if scope else '')
        else:
            return activation


def dice(_x, is_training, name=''):
    with tf.variable_scope(name_or_scope=name):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        beta = tf.get_variable('beta', _x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                               dtype=tf.float32)

    x_normed = tf.layers.batch_normalization(_x, center=False, scale=False, name=name, training=is_training)
    x_p = tf.sigmoid(beta * x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


def prelu(_x, scope=''):
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


class MaskNet:
    def __init__(self,
                 agg_dim: int,
                 num_mask_block: int,
                 mask_block_ffn_size: Union[List[int], int],
                 hidden_layer_size: Optional[List[int]] = None,
                 masknet_type: str = 'parallel',
                 dropout: float = 0.,
                 l2_reg: float = 0.
                 ):
        """

        :param agg_dim: Instance-Guided Mask中Aggregation模块的输出维度
        :param num_mask_block: 串行结构中MaskBlock的层数 or 并行结构中MaskBlock的数量
        :param mask_block_ffn_size: 每一层MaskBlock的输出维度
        :param masknet_type: serial(串行)或parallel(并行)
        :param hidden_layer_size: 并行结构中每一层隐藏层的输出维度
        :param dropout:
        :param l2_reg:
        """
        self.agg_dim = agg_dim
        self.num_mask_block = num_mask_block
        self.mask_block_ffn_size = mask_block_ffn_size
        self.hidden_layer_size = hidden_layer_size
        self.l2_reg = l2_reg

        self.dnn_layer = partial(dnn_layer, dropout=dropout, use_bn=False, l2_reg=l2_reg)

        if masknet_type == 'serial':
            self.net_func = self.serial_model
            assert isinstance(mask_block_ffn_size, list) and len(mask_block_ffn_size) == num_mask_block
        elif masknet_type == 'parallel':
            self.net_func = self.parallel_model
            assert isinstance(mask_block_ffn_size, int) and isinstance(hidden_layer_size, list)
        else:
            raise TypeError('masknet_type only support "serial" or "parallel"')

    def __call__(self,
                 embeddings: Union[List[tf.Tensor], tf.Tensor],
                 is_training: bool = True):
        """

        :param embeddings: [bs, num_feature, dim] or list of [bs, dim]
        :param is_training:
        :return:
        """
        # if isinstance(embeddings, list):
        #     embeddings = tf.stack(embeddings, axis=1)

        # assert len(embeddings.shape) == 3

        # ln_embeddings = tf.contrib.layers.layer_norm(inputs=embeddings,
        #                                              begin_norm_axis=-1,
        #                                              begin_params_axis=-1)
        ln_embeddings = embeddings
        # embeddings = tf.layers.flatten(embeddings)
        # ln_embeddings = tf.layers.flatten(ln_embeddings)

        output = self.net_func(embeddings, ln_embeddings, is_training)

        # output = tf.layers.dense(output, 1, activation=tf.nn.sigmoid,
        #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg),
        #                          kernel_initializer=tf.glorot_normal_initializer())
        # return tf.reshape(output, [-1])
        return output

    def serial_model(self, embeddings, ln_embeddings, is_training):
        """串行MaskNet"""
        output = ln_embeddings
        for i in range(self.num_mask_block):
            mask = self.instance_guided_mask(embeddings, is_training, output_size=output.shape.as_list()[-1])
            output = self.mask_block(output, mask, self.mask_block_ffn_size[i], is_training)

        return output

    def parallel_model(self, embeddings, ln_embeddings, is_training):
        """并行MaskNet"""
        output_list = []
        for i in range(self.num_mask_block):
            mask = self.instance_guided_mask(embeddings, is_training)
            output = self.mask_block(ln_embeddings, mask, self.mask_block_ffn_size, is_training)
            output_list.append(output)

        final_output = self.dnn_layer(tf.concat(output_list, axis=-1), self.hidden_layer_size, activation=tf.nn.relu,
                                      is_training=is_training)
        # final_output = tf.concat(output_list, axis=-1)
        # print("final::", final_output)
        return final_output

    def instance_guided_mask(self, embeddings, is_training, output_size=None):
        if output_size is None:
            output_size = embeddings.shape.as_list()[-1]
        agg = self.dnn_layer(embeddings, self.agg_dim, activation=tf.nn.relu, is_training=is_training)
        project = self.dnn_layer(agg, output_size, is_training=is_training)
        return project

    def mask_block(self, inputs, mask, output_size, is_training):
        masked = inputs * mask
        output = self.dnn_layer(masked, output_size, is_training=is_training)
        output = tf.contrib.layers.layer_norm(inputs=output,
                                              begin_norm_axis=-1,
                                              begin_params_axis=-1)
        return tf.nn.relu(output)