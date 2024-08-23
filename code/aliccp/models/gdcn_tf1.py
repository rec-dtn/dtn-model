# code from clabrugere 
# https://github.com/clabrugere/ctr-prediction/blob/8b58b49e975ca8dc3bdc41ec5449b696be522bef/models/tensorflow/gdcn.py
import tensorflow as tf
from tensorflow.keras import Model, activations
from tensorflow.keras.layers import Dense, Embedding, Layer
from functools import partial

from .mlp import MLP

# def dnn_layer(inputs: tf.Tensor,
#               hidden_units: Union[List[int], int],
#               activation: Optional[Union[Callable, str]] = None,
#               dropout: Optional[float] = 0.,
#               is_training: Optional[bool] = True,
#               use_bn: Optional[bool] = True,
#               l2_reg: float = 0.,
#               use_bias: bool = True,
#               scope=None):
#     if isinstance(hidden_units, int):
#         hidden_units = [hidden_units]

#     output = inputs
#     for idx, size in enumerate(hidden_units):
#         output = tf.layers.dense(output, size,
#                                  use_bias=use_bias,
#                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
#                                  kernel_initializer=tf.glorot_normal_initializer(),
#                                  name=scope+f'_{idx}' if scope else None)
#         if use_bn:
#             output = tf.layers.batch_normalization(output, training=is_training, name=scope+f'_bn_{idx}' if scope else None)

#         if activation is not None:
#             output = activation_layer(activation, is_training=is_training, scope=f'activation_layer_{idx}')(output)

#         if is_training:
#             output = tf.nn.dropout(output, 1 - dropout)

#     return output


# class GDCNS: 
#     def __init__(
#         self, 
#         dim_input,
#         num_embedding,
#         dim_embedding,
#         num_cross,
#         num_hidden,
#         dim_hidden,
#         dropout=0.0,
#         name="GDCN",
#         ): 
#         """ 
        
#         :param a
#         """
#         self.dim_input = dim_input 
#         self.num_embedding = num_embedding 
#         self.dim_embedding = dim_embedding
#         self.num_cross = num_cross
#         self.num_hidden = num_hidden 
#         self.dim_hidden = dim_hidden 
#         self.dropout = dropout 
#         self.name = name 



# class GDCNS(Model):
#     def __init__(
#         self,
#         dim_input,
#         num_embedding,
#         dim_embedding,
#         num_cross,
#         num_hidden,
#         dim_hidden,
#         dropout=0.0,
#         name="GDCN",
#     ):
#         super().__init__(name=name)
#         self.dim_input = dim_input
#         self.dim_embedding = dim_embedding

#         self.embedding = Embedding(
#             input_dim=num_embedding,
#             output_dim=dim_embedding,
#             name="embedding",
#         ) # 从id 到 embedding 索引 

#         self.cross = GatedCrossNetwork(num_cross)
#         self.projector = MLP(num_hidden, dim_hidden, dim_out=1, dropout=dropout)
#         self.build(input_shape=(None, dim_input))

#     def call(self, inputs, training=None):
#         out = self.embedding(inputs, training=training)
#         bottom = tf.reshape(out, (-1, self.dim_input * self.dim_embedding))
#         # bottom 是share bottom 底部  
#         out_embedding = self.cross(bottom, training=training)
        
#         # out = self.projector(out, training=training)
#         # out = activations.sigmoid(out)

#         return out_embedding

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

# class GDCNP(Model):
#     def __init__(
#         self,
#         dim_input,
#         num_embedding,
#         dim_embedding,
#         num_cross,
#         num_hidden,
#         dim_hidden,
#         dropout=0.0,
#         name="GDCN",
#     ):
#         super().__init__(name=name)
#         self.dim_input = dim_input
#         self.dim_embedding = dim_embedding

#         self.embedding = Embedding(
#             input_dim=num_embedding,
#             output_dim=dim_embedding,
#             name="embedding",
#         )

#         self.cross = GatedCrossNetwork(num_cross)
#         self.mlp = MLP(num_hidden, dim_hidden, dropout=dropout)
#         self.projector = Dense(1)
#         self.build(input_shape=(None, dim_input))

#     def call(self, inputs, training=None):
#         out = self.embedding(inputs, training=training)
#         out = tf.reshape(out, (-1, self.dim_input * self.dim_embedding))

#         out_1 = self.cross(out, training=training)
#         out_2 = self.mlp(out, training=training)

#         out_embedding = tf.concat((out_1, out_2), axis=-1)
#         # out = self.projector(out, training=training)
#         # out = tf.sigmoid(out)

#         return out_embedding


# class GatedCrossNetwork(Layer):
#     def __init__(
#         self, num_layers, weights_initializer="glorot_uniform", bias_initializer="zeros", name="GatedCrossNetwork"
#     ):
#         super().__init__(name=name)
#         self.num_layers = num_layers
#         self.weights_initializer = weights_initializer
#         self.bias_initializer = bias_initializer

#     def build(self, input_shape):
#         input_shape = tf.TensorShape(input_shape)
#         dim_input = input_shape[-1]

#         self._layers = []
#         for i in range(self.num_layers):
#             W_c = self.add_weight(
#                 name=f"W_c{i}",
#                 shape=(dim_input, dim_input),
#                 initializer=self.weights_initializer,
#                 dtype=self.dtype,
#                 trainable=True,
#             )
#             b_c = self.add_weight(
#                 name=f"b_c{i}",
#                 shape=(dim_input,),
#                 initializer=self.bias_initializer,
#                 dtype=self.dtype,
#                 trainable=True,
#             )
#             W_g = self.add_weight(
#                 name=f"W_g{i}",
#                 shape=(dim_input, dim_input),
#                 initializer=self.weights_initializer,
#                 dtype=self.dtype,
#                 trainable=True,
#             )
#             self._layers.append((W_c, b_c, W_g))

#     def call(self, inputs, training=None):
#         out = inputs  # (bs, dim_input)
#         for W_c, b_c, W_g in self._layers:
#             out = inputs * (tf.matmul(out, W_c) + b_c) * tf.sigmoid(tf.matmul(out, W_g)) + out  # (bs, dim_input)

#         return out