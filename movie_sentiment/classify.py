#!/usr/bin/env python2.7

import tensorflow as tf
import tensorflow_hub as hub
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
# import seaborn as sns

# Load all files from a directory in a DataFrame.
from movie_sentiment.accuracy import print_estimator_accuracy
from movie_sentiment.get_data import get_train_test_datasets


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
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))


def get_input_fn():
    sentences = [
        'This was an amazing film',
        'This movie is terrible',
        'Absolutely great',
        'Great film. I enjoyed and highly recommend it.',
        'Singing in the rain is a fabulously entertaining musical.',
        'Marchella is an awful example of humanity at its worst.'
    ]
    sentiment = [10, 1, 9, 9, 10, 1]
    polarity = [1, 0, 1, 1, 1, 0]
    data = {'sentence': sentences, 'sentiment': sentiment, 'polarity': polarity}
    df = pd.DataFrame(data)
    sentences_movies = list(zip(df['sentiment'].tolist()[:15], df['sentence'].tolist()[:15]))
    input_fn = tf.estimator.inputs.pandas_input_fn(df, df['polarity'], shuffle=False)
    return input_fn, sentences_movies


def classify(estimator_, classify_input_fn, train_input_fn, sentences_movies):
    estimates = estimator_.evaluate(input_fn=classify_input_fn)
    print 'Test set accuracy: {accuracy}'.format(**estimates)
    for est in estimates:
        print est

    def get_predictions(estimator_, input_fn):
        return [x['class_ids'][0] for x in estimator_.predict(input_fn=input_fn)]

    # predictions = estimator_.predict(input_fn=classify_input_fn)
    # print df
    # for idx, p in enumerate(predictions):
    # print p['logits'][0], p['probabilities'], p['classes'], p['classes'] == b'0', sentences_movies[idx]

    def print_res(estimator_x):
        for idx, p in enumerate(get_predictions(estimator_x, classify_input_fn)):
            print 'good' if p else 'bad', sentences_movies[idx]

    print_res(estimator_)

    def train_and_evaluate_with_module(hub_module, train_module=False):
        embedded_text_feature_column = hub.text_embedding_column(
            key='sentence', module_spec=hub_module, trainable=train_module)
        estimator = tf.estimator.DNNClassifier(
            hidden_units=[500, 100],
            feature_columns=[embedded_text_feature_column],
            n_classes=2,
            optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

        estimator.train(input_fn=train_input_fn, steps=1000)
        return estimator

    tf_modules = [
        ('https://tfhub.dev/google/nnlm-en-dim128/1', False),
        ('https://tfhub.dev/google/nnlm-en-dim128/1', True),
        ('https://tfhub.dev/google/random-nnlm-en-dim128/1', False),
        ('https://tfhub.dev/google/random-nnlm-en-dim128/1', True),
    ]
    for tfhub_module, train in tf_modules:
        print 'Running w/ module {}; trained={}'.format(tfhub_module, train)
        print_res(train_and_evaluate_with_module(tfhub_module, train))


def main():
    train_df, test_df = get_train_test_datasets()
    estimator = get_estimator(train_df=train_df, test_df=test_df)
    print_estimator_accuracy(estimator, train_df=train_df, test_df=test_df)

    input_fn, sentences_movies = get_input_fn()

    # Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df['polarity'], num_epochs=None, shuffle=True)

    classify(estimator, sentences_movies=sentences_movies, classify_input_fn=input_fn, train_input_fn=train_input_fn)


if __name__ == '__main__':
    main()