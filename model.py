import tensorflow as tf
from hparams import hparams
from modules import get_next_batch
from networks import lstm_3_layers
hp = hparams()
class Graph:
    def __init__(self, mode='train'):
        self.mode = mode
        self.scope_name = 'net'
        self.reuse = tf.AUTO_REUSE

    def train(self):
        self.x, self.y = get_next_batch()
        self.global_step = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
        self.lr = tf.train.exponential_decay(learning_rate=hp.LR, global_step=self.global_step,
                                             decay_rate=hp.DECAY_RATE,
                                             decay_steps=hp.DECAY_STEPS)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        with tf.variable_scope(self.scope_name, reuse=self.reuse):
            self.y_hat = lstm_3_layers(self.x, num_units=hp.UNITS, bidirection=False, scope='lstm_3_layers') # [N, U]
            self.y_hat = tf.layers.dense(self.y_hat, units=hp.LABEL_SIZE, activation=tf.nn.sigmoid, scope='dense_1') # [N, L]
        self.loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.y - self.y_hat)))
        self.grads = self.optimizer.compute_gradients(self.loss)
        clipped = []
        for grad, var in self.grads:
            grad = tf.clip_by_norm(grad, 5.)
            clipped.append((grad, var))
        self.train_op = self.optimizer.apply_gradients(clipped, global_step=self.global_step)

    def infer(self):
        self.x = tf.placeholder([None, None, hp.FEATURE_SIZE])
        with tf.variable_scope(self.scope_name, reuse=self.reuse):
            self.y_hat = lstm_3_layers(self.x, num_units=hp.UNITS, bidirection=False, scope='lstm_3_layers')
            self.y_hat = tf.layers.dense(self.y_hat, units=hp.LABEL_SIZE, activation=tf.nn.sigmoid, scope='dense_1')
