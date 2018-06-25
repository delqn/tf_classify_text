import tensorflow_hub as hub


def get_estimator(train_df, test_df):
    """Estimator: for classification we can use a DNN Classifier"""
    # Feature columns: TF-Hub provides a feature column that applies a module on the
    # given text feature and passes further the outputs of the module.
    embedded_text_feature_column = hub.text_embedding_column(
        key='sentence',
        module_spec='https://tfhub.dev/google/nnlm-en-dim128/1')
    return tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
        model_dir='./models',
    )