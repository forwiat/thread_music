import tensorflow as tf
import glob
from hparams import hparams
hp = hparams()
def get_next_batch():
    tfrecords = glob.glob(f'{hp.TF_DIR}/*.tfrecord')
    filename_queue = tf.train.string_input_producer(tfrecords, shuffle=True, num_epochs=hp.NUM_EPOCHS)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
            'x': tf.FixedLenFeature([], tf.float32),
            'y': tf.FixedLenFeature([], tf.float32)
        }
    )
    x = tf.reshape(features['x'], [hp.SEGMENT, hp.FEATURE_SIZE])
    y = tf.reshape(features['y'], [hp.LABEL_SIZE])
    x_batch, y_batch = tf.train.shuffle_batch([x, y], batch_size=hp.BATCH_SIZE, capacity=100,
                                              min_after_dequeue=hp.BATCH_SIZE*5, num_threads=10)
    return x_batch, y_batch
