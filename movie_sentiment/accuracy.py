def print_estimator_accuracy(estimator, train_df, test_df):
    """Training input on the whole training set with no limit on training epochs."""
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df['polarity'], num_epochs=None, shuffle=True)

    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df['polarity'], shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df['polarity'], shuffle=False)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print 'Training set accuracy: {accuracy}'.format(**train_eval_result)
    print 'Test set accuracy: {accuracy}'.format(**test_eval_result)