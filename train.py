from model import Graph
import tensorflow as tf
from hparams import hparams
import os
hp = hparams()
def main():
    mode = 'train'
    G = Graph(mode=mode)
    print('{} graph loaded.'.format(mode))
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(hp.LOG_DIR, sess.graph)
        try:
            print(f'Try to load trained model in {hp.MODEL_DIR} ...')
            saver.restore(sess, tf.train.latest_checkpoint(hp.MODEL_DIR))
        except:
            print('Load trained model failed, start training with initializer ...')
            sess.run(tf.global_variables_initializer())
        finally:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while not coord.should_stop():
                    _, loss, summary, steps = sess.run([G.train_op, G.loss, G.merged, G.global_step])
                    print('train mode \t steps : {} \t loss : {}'.format(steps, loss))
                    writer.add_summary(summary=summary, global_step=steps)
                    if steps % (hp.PER_STEPS + 1) == 0:
                        saver.save(sess, os.path.join(hp.MODEL_DIR, 'model_{}los_{}steps'.format(loss, steps)))
            except tf.errors.OutOfRangeError:
                print('Training Done.')
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    main()
