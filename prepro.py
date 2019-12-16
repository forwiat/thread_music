import tensorflow as tf
from utils import get_spectrogram
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import codecs
import os
from hparams import hparams
hp = hparams()

def thread_process(args):
    (tfid, split_dataset) = args
    writer = tf.python_io.TFRecordWriter(os.path.join(hp.TF_DIR, f'{tfid}.tfrecord'))
    for i in tqdm(split_dataset):
        fpath = i[0]
        y = i[1]
        x = get_spectrogram(fpath)
        example = tf.train.Example(features=tf.train.Features(feature={
            'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.reshape(-1))),
            'y': tf.train.Feature(float_list=tf.train.FloatList(value=y.reshape(-1)))
        }))
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def preprocess():
    lines = codecs.open(hp.CSV_PATH, 'r').readlines()
    dataset = []
    for line in lines:
        fpath, label_str = line.strip().split('\t')
        labels = label_str.strip().split('|')
        y = np.zeros(shape=[hp.LABEL_SIZE], dtype=np.float32)
        for j in [hp.LABEL_DIC[i] for i in labels]:
            y[j] = 1
        dataset.append([fpath, y])
    if hp.MULTI_PROCESS:
        cpu_nums = mp.cpu_count()
        thread_nums = int(cpu_nums * hp.CPU_RATE)
        splits = [(i, dataset[i::thread_nums])
                  for i in range(thread_nums)]
        pool = mp.Pool(thread_nums)
        pool.map(thread_process, splits)
        pool.close()
        pool.join()
    else:
        splits = (0, dataset)
        thread_process(splits)

def check_paths():
    if os.path.exists(hp.TF_DIR) is False:
        os.makedirs(hp.TF_DIR)

if __name__ == '__main__':
    check_paths()
    preprocess()
