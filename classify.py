#!/usr/bin/env python2.7

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

# Load all files from a directory in a DataFrame.
def load_as_dataframe(directory):
  data = {
      'sentence': [],
      'sentiment': [],
  }
  for file_path in os.listdir(directory):
    with tf.gfile.GFile(os.path.join(directory, file_path), 'r') as f:
      data['sentence'].append(f.read())
      data['sentiment'].append(re.match('\d+_(\d+)\.txt', file_path).group(1))
  return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_as_dataframe(os.path.join(directory, 'pos'))
  neg_df = load_as_dataframe(os.path.join(directory, 'neg'))
  pos_df['polarity'] = 1
  neg_df['polarity'] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def download_and_load_datasets(force_download=False):
    """Download and process the dataset files."""
    origin = './aclImdb_v1.tar.gz'
    if force_download:
        origin = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    print('Loading dataset from {}'.format(origin))
    dataset = tf.keras.utils.get_file(fname='aclImdb.tar.gz', origin=origin, extract=True)
    print('Loaded dataset -> {}: {}'.format(type(dataset), dataset))
    return (
        load_dataset(os.path.join(os.path.dirname(dataset), 'aclImdb', 'train')),
        load_dataset(os.path.join(os.path.dirname(dataset), 'aclImdb', 'test')))


def train():
    tf.logging.set_verbosity(tf.logging.INFO)
    train_df, test_df = download_and_load_datasets()
    train_df.head()
    return train_df, test_df


def get_estimator(train_df, test_df):
    """Training input on the whole training set with no limit on training epochs."""
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df['polarity'], num_epochs=None, shuffle=True)

    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df['polarity'], shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df['polarity'], shuffle=False)

    # Feature columns: TF-Hub provides a feature column that applies a module on the
    # given text feature and passes further the outputs of the module.
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    # Estimator: for classification we can use a DNN Classifier
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print "Training set accuracy: {accuracy}".format(**train_eval_result)
    print "Test set accuracy: {accuracy}".format(**test_eval_result)
    return estimator


def hacked_out(estimator):
    sentences = [
        'This was an amazing film',
        'This movie is terrible',
        'Absolutely great',
        'Ricky',
    ]
    data = {'sentence': sentences}
    df = pd.DataFrame(data)

    df = train_df

    ########

    sentences_movies = df['sentence'].tolist()[:15]


    input_fn = tf.estimator.inputs.pandas_input_fn(
        df,
        train_df["polarity"],
        shuffle=False)

    estimates = estimator.evaluate(input_fn=input_fn)
    print("Test set accuracy: {accuracy}".format(**estimates))
    for est in estimates:
        print(est)


        predictions = estimator.predict(input_fn=input_fn)
        print(df)
        for idx, p in enumerate(predictions):
            print(p['logits'][0], p['probabilities'], p['classes'], p['classes'] == b'0', sentences_movies[idx])


def hacked_out_1(estimator):
    sentences = [
        'This was an amazing film',
        'This movie is terrible',
        'Absolutely great',
        'Great film. I enjoyed and highly recommend it.',
        'Singing in the rain is a fabulously entertaining musical.',
        'Marchella is an awful example of humanity at its worst.'
    ]
    sentiment = [10, 1, 9, 9, 0, 0]
    polarity = [1,0,1,1, 0, 0]
    data = {'sentence': sentences, 'sentiment': sentiment, 'polarity': polarity}
    df = pd.DataFrame(data)

    # df = train_df

    ########

    sentences_movies = list(zip(df['sentiment'].tolist()[:15], df['sentence'].tolist()[:15]))


    input_fn = tf.estimator.inputs.pandas_input_fn(
        df,
        df["polarity"],
        shuffle=False)

    estimates = estimator.evaluate(input_fn=input_fn)
    print("Test set accuracy: {accuracy}".format(**estimates))
    for est in estimates:
      print(est)

    def get_predictions(estimator, input_fn):
      return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

    predictions = estimator.predict(input_fn=input_fn)
    ## print(df)
    # for idx, p in enumerate(predictions):
    #  print(p['logits'][0], p['probabilities'], p['classes'], p['classes'] == b'0', sentences_movies[idx])

    for idx, p in enumerate(get_predictions(estimator, input_fn)):
      print('good' if p else 'bad', sentences_movies[idx])

def hacked_out_2():
    sentences = [
        'This was an amazing film',
        'This movie is terrible',
        'Absolutely great',
        'Great film. I enjoyed and highly recommend it.',
        'Singing in the rain is a fabulously entertaining musical.',
        'Marchella is an awful example of humanity at its worst.'
    ]
    sentiment = [10, 1, 9, 9, 0, 0]
    polarity = [1,0,1,1, 0, 0]
    data = {'sentence': sentences, 'sentiment': sentiment, 'polarity': polarity}
    df = pd.DataFrame(data)

    # df = train_df

    ########

    sentences_movies = list(zip(df['sentiment'].tolist()[:15], df['sentence'].tolist()[:15]))


    input_fn = tf.estimator.inputs.pandas_input_fn(
        df,
        df["polarity"],
        shuffle=False)

    estimates = estimator.evaluate(input_fn=input_fn)
    print("Test set accuracy: {accuracy}".format(**estimates))
    for est in estimates:
      print(est)

    def get_predictions(estimator, input_fn):
      return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

    predictions = estimator.predict(input_fn=input_fn)
    ## print(df)
    # for idx, p in enumerate(predictions):
    #  print(p['logits'][0], p['probabilities'], p['classes'], p['classes'] == b'0', sentences_movies[idx])

    def print_res(estimator_):
        for idx, p in enumerate(get_predictions(estimator_, input_fn)):
            print('good' if p else 'bad', sentences_movies[idx])


    def train_and_evaluate_with_module(hub_module, train_module=False):
      embedded_text_feature_column = hub.text_embedding_column(
          key="sentence", module_spec=hub_module, trainable=train_module)

      estimator = tf.estimator.DNNClassifier(
          hidden_units=[500, 100],
          feature_columns=[embedded_text_feature_column],
          n_classes=2,
          optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

      estimator.train(input_fn=train_input_fn, steps=1000)
      return estimator

    print_res(train_and_evaluate_with_module("https://tfhub.dev/google/nnlm-en-dim128/1"))
    print_res(train_and_evaluate_with_module("https://tfhub.dev/google/nnlm-en-dim128/1", True))
    print_res(train_and_evaluate_with_module("https://tfhub.dev/google/random-nnlm-en-dim128/1"))
    print_res(train_and_evaluate_with_module("https://tfhub.dev/google/random-nnlm-en-dim128/1", True))


if __name__ == '__main__':
    hacked_out_1(get_estimator(*train()))
    hacked_out_2(get_estimator(*train()))
