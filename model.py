import tensorflow as tf
import numpy as np
from hparams import hparams
from modules import get_next_batch
from networks import lstm_3_layers
hp = hparams()
class Graph:
    def __init__(self, mode='train'):
        self.mode = mode
        self.scope_name = 'net'
        self.reuse = tf.AUTO_REUSE
        if self.mode in ['train', 'eval']:
            if self.mode == 'train' and len(hp.GPU_IDS) > 1:
                self.multi_train()
            else:
                self.single_train()
            tf.summary.scalar('{}/loss'.format(self.mode), self.loss)
            self.merged = tf.summary.merge_all()
            self.t_vars = tf.trainable_variables()
            self.num_paras = 0
            for var in self.t_vars:
                var_shape = var.get_shape().as_list()
                self.num_paras += np.prod(var_shape)
            print("Total number of parameters : %r" % self.num_paras)
        elif self.mode in ['test']:
            self.test()
        elif self.mode in ['infer']:
            self.infer()
        else:
            raise Exception('No supported mode in model __init__ function, please check ...')
    ###################################################################################
    #                                                                                 #
    #                            single gpu train and eval                            #
    #                                                                                 #
    ###################################################################################

    def single_train(self):
        self.x, self.y = get_next_batch()
        self.global_step = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
        self.lr = tf.train.exponential_decay(learning_rate=hp.LR, global_step=self.global_step,
                                             decay_rate=hp.DECAY_RATE,
                                             decay_steps=hp.DECAY_STEPS)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        with tf.variable_scope(self.scope_name, reuse=self.reuse):
            self.y_hat = lstm_3_layers(self.x, num_units=hp.UNITS, bidirection=False, scope='lstm_3_layers') # [N, U]
            self.y_hat = tf.layers.dense(self.y_hat, units=hp.LABEL_SIZE*2, activation=tf.nn.tanh, name='dense_1') # [N, L*2]
            self.y_hat = tf.layers.dense(self.y_hat, units=hp.LABEL_SIZE, activation=tf.nn.sigmoid, name='output_1') # [N, L]
        self.loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.y - self.y_hat)))
        self.grads = self.optimizer.compute_gradients(self.loss)
        clipped = []
        for grad, var in self.grads:
            grad = tf.clip_by_norm(grad, 5.)
            clipped.append((grad, var))
        self.train_op = self.optimizer.apply_gradients(clipped, global_step=self.global_step)

    ###################################################################################
    #                                                                                 #
    #                                   multi gpu train                               #
    #                                                                                 #
    ###################################################################################

    def multi_train(self):
        def _assign_to_device(device, ps_device='/cpu:0'):
            PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
            def _assign(op):
                node_def = op if isinstance(op, tf.NodeDef) else op.node_def
                if node_def.op in PS_OPS:
                    return '/' + ps_device
                else:
                    return device
            return _assign
        def _average_gradients(tower_grads):
            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g, 0)
                    grads.append(expanded_g)
                grad = tf.concat(grads, 0)
                grad = tf.reduce_mean(grad, 0)
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)
                average_grads.append(grad_and_var)
            return average_grads
        with tf.device('/cpu:0'):
            self.x, self.y, self.mask = get_next_batch()
            self.tower_grads = []
            self.global_step = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
            self.lr = tf.train.exponential_decay(hp.LR, global_step=self.global_step,
                                                 decay_steps=hp.DECAY_STEPS,
                                                 decay_rate=hp.DECAY_RATE)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            gpu_nums = len(hp.GPU_IDS)
            per_batch = hp.BATCH_SIZE // gpu_nums
            with tf.variable_scope(self.scope_name, reuse=self.reuse):
                for i in range(gpu_nums):
                    with tf.device(_assign_to_device('/gpu:{}'.format(hp.GPU_IDS[i]), ps_device='/cpu:0')):
                        self._x = self.x[i * per_batch: (i + 1) * per_batch]
                        self._y = self.y[i * per_batch: (i + 1) * per_batch]
                        self._mask = self.mask[i * per_batch: (i + 1) * per_batch]
                        self.y_hat = lstm_3_layers(self.x, num_units=hp.UNITS, bidirection=False, scope='lstm_3_layers') # [N, U]
                        self.y_hat = tf.layers.dense(self.y_hat, units=hp.LABEL_SIZE*2, activation=tf.nn.tanh, name='dense_1') # [N, L*2]
                        self.y_hat = tf.layers.dense(self.y_hat, units=hp.LABEL_SIZE, activation=tf.nn.sigmoid, name='output_1') # [N, L]
                        tf.get_variable_scope().reuse_variables()
                        # loss
                        self.loss = tf.reduce_sum(tf.reduce_mean(tf.square(self.y - self.y_hat)))
                        self.grads = self.optimizer.compute_gradients(self.loss)
                        self.tower_grads.append(self.grads)
            self.tower_grads = _average_gradients(self.tower_grads)
            clipped = []
            for grad, var in self.tower_grads:
                grad = tf.clip_by_norm(grad, 5.)
                clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(clipped, global_step=self.global_step)

    ###################################################################################
    #                                                                                 #
    #                                  test data in cpu                               #
    #                                                                                 #
    ###################################################################################

    def test(self):
        with tf.device('/cpu:0'):
            self.x, self.y = get_next_batch()
            with tf.variable_scope(self.scope_name, reuse=self.reuse):
                self.y_hat = lstm_3_layers(self.x, num_units=hp.UNITS, bidirection=False, scope='lstm_3_layers') # [N, U]
                self.y_hat = tf.layers.dense(self.y_hat, units=hp.LABEL_SIZE*2, activation=tf.nn.tanh, name='dense_1') # [N, L*2]
                self.y_hat = tf.layers.dense(self.y_hat, units=hp.LABEL_SIZE, activation=tf.nn.sigmoid, name='output_1') # [N, L]

    ###################################################################################
    #                                                                                 #
    #                             real data infer in cpu                              #
    #                                                                                 #
    ###################################################################################

    def infer(self):
        with tf.device('/cpu:0'):
            self.x = tf.placeholder([None, None, hp.FEATURE_SIZE])
            with tf.variable_scope(self.scope_name, reuse=self.reuse):
                self.y_hat = lstm_3_layers(self.x, num_units=hp.UNITS, bidirection=False, scope='lstm_3_layers') # [N, U]
                self.y_hat = tf.layers.dense(self.y_hat, units=hp.LABEL_SIZE*2, activation=tf.nn.tanh, name='dense_1') # [N, L*2]
                self.y_hat = tf.layers.dense(self.y_hat, units=hp.LABEL_SIZE, activation=tf.nn.sigmoid, name='output_1') # [N, L]
