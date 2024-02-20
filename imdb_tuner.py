import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
  
LABEL_KEY = "sentiment"
FEATURE_KEY = "review"
NUM_EPOCHS = 5
# Vocabulary size and number of words in a sequence.
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 200
vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"
  
def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
  
  
def input_fn(file_pattern, 
              tf_transform_output,
              num_epochs,
              batch_size=64)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transformed_name(LABEL_KEY))
    return dataset
  
# os.environ['TFHUB_CACHE_DIR'] = '/hub_chace'
# embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")
  


def model_builder(hp):
    """Build machine learning model"""
    num_hidden_layers = hp.Choice(
        "num_hidden_layers", values=[1, 2]
    )
    embed_dims = hp.Int(
        "embed_dims", min_value=16, max_value=128, step=32
    )
    dense_units = hp.Int(
        "dense_units", min_value=32, max_value=256, step=32
    )
    learning_rate = hp.Choice(
        "learning_rate", values=[1e-2, 1e-3, 1e-4]
    )
    lstm_units= hp.Int(
        "lstm_units", min_value=32, max_value=128, step=32
    )
    dropout_rate = hp.Float(
        "dropout_rate", min_value=0.1, max_value=0.5, step=0.1
    )

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=embed_dims)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)

    for _ in range(num_hidden_layers):
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics='binary_accuracy'
    )
    return model
  
def tuner_fn(fn_args):
    """Build the tuner using the KerasTuner API.
    Args:
    fn_args: Holds args used to tune models as name/value pairs.
    
    Returns:
    A namedtuple contains the following:
        - tuner: A BaseTuner that will be used for tuning.
        - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
    """
    # Memuat training dan validation dataset yang telah di-preprocessing
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, NUM_EPOCHS)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS)

    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)]])
    # Mendefinisikan strategi hyperparameter tuning
    tuner = kt.RandomSearch(hypermodel=lambda hp: model_builder(hp),
                        objective='val_binary_accuracy',
                        max_trials=5,
                        directory=fn_args.working_dir,
                        project_name="kt_randomsearch",
                        )
    # Define the early stopping callback
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
    return TunerFnResult(
      tuner=tuner,
      fit_kwargs={ 
          "callbacks":[stop_early],
          'x': train_set,
          'validation_data': val_set,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      }
  )
