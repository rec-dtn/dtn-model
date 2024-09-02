import tensorflow as tf
from models.components.inputs import SparseFeat, DenseFeat, VarLenSparseFeat

def get_top_inputs_embeddings(feature_columns, features, embeddings, feature_importance_metric="dimension",
                              feature_importance_top_k=-1, return_feature_index=False):
    """
    Get top K inputs and embeddings following importance metric
    
    Note: the order of features and embeddings is different
    :param feature_columns: feature columns in dataset
    :param features: dict, key is feature nane, value is the Input of a feature to a model
    :param embeddings: embeddings of features
    :param feature_importance_metric: metric of feature importance
    :param feature_importance_top_k: top K
    :param return_feature_index: boolean
    :return:
    """
    print("########## get top check ##########")
    # Default use all inputs and embeddings
    if feature_importance_top_k == -1:
        feature_importance_top_k = len(feature_columns)

    # Get the TopK most important feature names
    sorted_feature_columns = []
    for fc in feature_columns:
        sorted_feature_columns.append((fc.name, getattr(fc, feature_importance_metric)))
        
    sorted_feature_columns.sort(key=lambda f: f[1], reverse=True)
    selected_feature_columns = set([sorted_feature_columns[i][0] for i in range(feature_importance_top_k)])

    # Get the TopK most important feature inputs in order of [sparse inputs, var len inputs, dense inputs]
    top_inputs = []
    for idx, fc in enumerate(feature_columns): 
        if fc.name in selected_feature_columns: 
            top_inputs.append(tf.expand_dims(fc.input, axis=1))
    
    count_sparse_features = 0
    offsets_sparse_features = []
    for idx, fc in enumerate(feature_columns): 
        if fc.name in selected_feature_columns: 
            offsets_sparse_features.append(count_sparse_features) 
        count_sparse_features += 1 
    # embeddings = [sparse_embeddings, var_len_embeddings, dense_embeddings]
    selected_features_indexes = [idx for idx in offsets_sparse_features]
    top_embeddings = tf.gather(embeddings, selected_features_indexes, axis=1)
    print("[INFO]: select: ", top_inputs, top_embeddings)
    if return_feature_index:
        return top_inputs, top_embeddings, selected_features_indexes
    return top_inputs, top_embeddings
