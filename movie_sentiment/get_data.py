import os
import re


def load_as_dataframe(directory):
    data = {
        'sentence': [],
        'sentiment': [],
    }
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), 'r') as f:
            data['sentence'].append(f.read())
            data['sentiment'].append(re.match(r'\d+_(\d+)\.txt', file_path).group(1))
    return pd.DataFrame.from_dict(data)


def file_to_dataframe(directory):
    """Merge positive and negative examples, add a polarity column and shuffle."""
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
    print 'Loading dataset from {}'.format(origin)
    dataset = tf.keras.utils.get_file(fname='aclImdb.tar.gz', origin=origin, extract=True)
    print 'Loaded dataset -> {}: {}'.format(type(dataset), dataset)
    return (
        file_to_dataframe(os.path.join(os.path.dirname(dataset), 'aclImdb', 'train')),
        file_to_dataframe(os.path.join(os.path.dirname(dataset), 'aclImdb', 'test')))


def get_train_test_datasets():
    tf.logging.set_verbosity(tf.logging.INFO)
    train_df, test_df = download_and_load_datasets()
    train_df.head()
    return train_df, test_df