from model import Graph
import tensorflow as tf
def main():
    G = Graph(mode='train')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while True:
                _, loss, steps = sess.run([G.train_op, G.loss, G.global_step])
        except:
            print('Training Done.')
        finally:
            coord.request_stop()
        coord.join(threads)
