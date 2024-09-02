# code from clabrugere 
# https://github.com/clabrugere/ctr-prediction/blob/8b58b49e975ca8dc3bdc41ec5449b696be522bef/models/tensorflow/gdcn.py
import tensorflow as tf
from tensorflow.keras import Model, activations
from tensorflow.keras.layers import Dense, Embedding, Layer
from functools import partial

class GDCN_net: 
    def __init__( 
        self, 
        task_idx, 
        num_layers, 
        output_dim, 
        input_shape, 
        weights_initializer = tf.glorot_uniform_initializer, 
        bias_initializer = tf.zeros_initializer,
        dtype = tf.float32,
        name="GatedCrossNetwork"
    ):
        self.task_idx = task_idx
        self.num_layers = num_layers 
        self.output_dim = output_dim
        self.input_shape = input_shape 
        self.weights_initializer = weights_initializer 
        self.bias_initializer = bias_initializer 
        self.dtype = dtype 

        dim_input = input_shape[-1] 
        self._layers = []
        for i in range(self.num_layers): 
            W_c = tf.get_variable(
                name = f"{task_idx}_W_c{i}",
                shape = (dim_input, dim_input), 
                initializer=self.weights_initializer, 
                dtype=self.dtype, 
                trainable=True 
            )
            b_c = tf.get_variable(
                name = f"{task_idx}_b_c{i}", 
                shape = (dim_input, ), 
                initializer=self.bias_initializer, 
                dtype = self.dtype,  
                trainable = True 
            )
            W_g = tf.get_variable( 
                name = f"{task_idx}_W_g{i}", 
                shape = (dim_input, dim_input), 
                initializer=self.weights_initializer, 
                dtype=self.dtype, 
                trainable=True 
            )
            self._layers.append((W_c, b_c, W_g))
        # self.dnn_layer = partial(dnn_layer, dropout=0, use_bn=False, l2_reg=l2_reg)

    def __call__(self, inputs, training=None):
        out = inputs # (bs, dim_input) 
        for W_c, b_c, W_g in self._layers: 
            out = inputs * (tf.matmul(out, W_c) + b_c) * tf.sigmoid(tf.matmul(out, W_g)) + out # (bs, dim_input) 
        
        # final_output = self.dnn_layer(tf.concat(output_list, axis=-1), self.hidden_layer_size, activation=tf.nn.relu,
        #         is_training=is_training)
        final_output = tf.layers.dense(out, self.output_dim,
                    use_bias=True,
                    # kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                    kernel_initializer=tf.glorot_normal_initializer(),
                    name=f'{self.task_idx}')
        return final_output
