import tensorflow as tf
import tensorflow_transform as tft
LABEL_KEY = "sentiment"
FEATURE_KEY = "review"
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"
def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    """
    # Define the number of buckets for hashing
    sentiment_vocab = ['negative', 'positive']
    # temp = inputs['sentiment']
    
    # Create a vocabulary for sentiment labels
    sentiment_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(sentiment_vocab, tf.constant(list(range(len(sentiment_vocab))), dtype=tf.int64)),
        default_value=tf.constant(-1, dtype=tf.int64),
    )
    
    # Map sentiment strings to numerical labels
    inputs['sentiment'] = sentiment_table.lookup(inputs['sentiment'])

    
    outputs = {}
    
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    # tf.print("Record data:", temp, "label: ", outputs['sentiment_xf'])
    # tf.print("Record data:", outputs['sentiment_xf'])
    
    return outputs
