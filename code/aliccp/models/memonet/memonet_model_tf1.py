"""
Code: https://github.com/ptzhangAlg/RecAlg/blob/master/README.md
"""
# -*- coding:utf-8 -*-
import copy

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model

from models.common import tf_utils
from models.common.utils import Utils
from models.components.inputs import build_input_features, input_from_feature_columns
from models.components.layers import DenseEmbeddingLayer, PredictionLayer, DNNLayer
from models.components.multi_hash_codebook_layer import MultiHashCodebookLayer
from models.components.multi_hash_codebook_kif_layer import MultiHashCodebookKIFLayer

class MemoNetModel_tf1(Model):
    def __init__(self, task_idx, feature_columns, input, feat_input_list, feat_emb_list, embedding_size, params = dict(), 
                output_dim = 256, embedding_l2_reg=0.0, embedding_dropout=0,
                dnn_hidden_units=(), dnn_activation='relu', dnn_l2_reg=0.0, dnn_use_bn=False,
                dnn_dropout=0.0, init_std=0.01, task='binary', seed=2021):
        super(MemoNetModel_tf1, self).__init__()
        self.task_idx = task_idx
        self.feature_columns = feature_columns 
        self.feat_input_list = feat_input_list 
        self.feat_emb_list = feat_emb_list
        self.field_size = len(feat_input_list) 
        self.params = params

        self.embedding_size = embedding_size
        self.output_dim = output_dim
        self.embedding_l2_reg = embedding_l2_reg
        self.embedding_dropout = embedding_dropout

        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activation = dnn_activation
        self.dnn_l2_reg = dnn_l2_reg
        self.dnn_use_bn = dnn_use_bn
        self.dnn_dropout = dnn_dropout

        self.init_std = init_std
        self.task = task
        self.seed = seed

        self.interact_mode = self.params.get("interact_mode", "fullhcnet")
        self.interaction_hash_embedding_buckets = self.params.get("interaction_hash_embedding_buckets", 100000)
        self.interaction_hash_embedding_size = self.params.get("interaction_hash_embedding_size", self.embedding_size)
        self.interaction_hash_embedding_bucket_mode = self.params.get("interaction_hash_embedding_bucket_mode", "hash-share")
        self.interaction_hash_embedding_num_hash = self.params.get("interaction_hash_embedding_num_hash", 2)
        self.interaction_hash_embedding_merge_mode = self.params.get("interaction_hash_embedding_merge_mode", "concat")
        self.interaction_hash_output_dims = self.params.get("interaction_hash_output_dims", 0)
        self.interaction_hash_embedding_float_precision = self.params.get("interaction_hash_embedding_float_precision", 12)
        self.interaction_hash_embedding_interact_orders = self.params.get("interaction_hash_embedding_interact_orders", (2,))
        self.interaction_hash_embedding_interact_modes = self.params.get("interaction_hash_embedding_interact_modes", ("none",))
        self.interaction_hash_embedding_feature_metric = self.params.get("interaction_hash_embedding_feature_metric", "dimension")
        self.interaction_hash_embedding_feature_top_k = self.params.get("interaction_hash_embedding_feature_top_k", -1)

    def  __build__(self):
        print("start buildding")
        # Initialize layers here
        self.multi_hash_codebook_layer = MultiHashCodebookLayer(
            name="multi_hash_codebook_layer",
            num_buckets=self.interaction_hash_embedding_buckets,
            embedding_size=self.interaction_hash_embedding_size,
            bucket_mode=self.interaction_hash_embedding_bucket_mode,
            init_std=self.init_std,
            l2_reg=self.embedding_l2_reg,
            seed=self.seed,
            num_hash=self.interaction_hash_embedding_num_hash,
            merge_mode=self.interaction_hash_embedding_merge_mode,
            output_dims=self.interaction_hash_output_dims,
            params=self.params,
            hash_float_precision=self.interaction_hash_embedding_float_precision,
            interact_orders=self.interaction_hash_embedding_interact_orders,
            interact_modes=self.interaction_hash_embedding_interact_modes,
        )
        self.dnn_layers = [Dense(units, activation=self.dnn_activation, kernel_regularizer=tf.keras.regularizers.l2(self.dnn_l2_reg)) 
                           for units in self.dnn_hidden_units]
        if self.dnn_use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in self.dnn_hidden_units]
        # self.dropout_layers = [Dropout(self.dnn_dropout) for _ in self.dnn_hidden_units]
        # self.final_dense = Dense(1, use_bias=True, activation=None)
        # self.prediction_layer = PredictionLayer(self.task, use_bias=False)
    
    def __call__(self):
        self.__build__()
        # features = build_input_features(self.feature_columns)
        # embeddings = self.get_embeddings(features)
        features = self.feat_input_list 
        embeddings = self.feat_emb_list 
        interact_embeddings = [embeddings]
        
        if "fullhcnet" in self.interact_mode:
            top_inputs_list, top_embeddings = tf_utils.get_top_inputs_embeddings(
                feature_columns=self.feature_columns, features=features, embeddings=embeddings,
                feature_importance_metric=self.interaction_hash_embedding_feature_metric,
                feature_importance_top_k=self.interaction_hash_embedding_feature_top_k)
            top_inputs_list = [tf.strings.to_number(x, out_type=tf.float32) for x in top_inputs_list]
            interaction_hash_embeddings, interact_field_weights = self.multi_hash_codebook_layer(
                [top_inputs_list, top_embeddings])
            interact_embeddings.append(interaction_hash_embeddings)

        interact_embeddings = [Flatten()(emb) for emb in interact_embeddings]
        concat_embedding = Utils.concat_func(interact_embeddings, axis=1)
        final_output = tf.layers.dense(concat_embedding, self.output_dim,
                use_bias=True,
                # kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                kernel_initializer=tf.glorot_normal_initializer(),
                name=f'{self.task_idx}_dense')
        return final_output 

