#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import string
#import json
import random
import math
import sys
# from pyspark.context import SparkContext
# from pyspark.conf import SparkConf
# import mysql.connector
# from mysql.connector import errorcode
from .utils import image_util
from .utils import mysql_util
from .utils import tf_util
from .utils import dataset_util
import tensorflow as tf

tf.app.flags.DEFINE_float('cvt_percent_validation', 0, 'Percent of images in the validation set.')
tf.app.flags.DEFINE_integer('cvt_random_seed', 0, 'Number of seed for repeatability.')
tf.app.flags.DEFINE_integer('cvt_num_shards', 2, 'Number of shards per dataset split.')
tf.app.flags.DEFINE_string('cvt_path_prefix', '/tmp/images', 'The path of images has been converted.')
FLAGS = tf.app.flags.FLAGS


# The number of images in the validation set.
#_NUM_VALIDATION = 20
# Seed for repeatability.
#_RANDOM_SEED = 0
# The number of shards per dataset split.
#_NUM_SHARDS = 2
# 图片存放的目录
#_PATH_PRFIX="/data/jg/image/images/"

train_no = None


def _get_train_no():
    return train_no


def _set_train_no(text):
    global train_no
    train_no = text


def _get_urls_and_classes(_train_no):
    class_names = []
    id_name_urls = []
    classifications = mysql_util.get_classification(_train_no)
    class_ids = []
    class_kvs = {}
    for _no, _id, _name in classifications:
        class_ids.append(_id)
        class_names.append(_name)
        class_kvs[_id] = _name

    classification_urls = mysql_util.get_classification_urls(class_ids)
    for classification_id, url in classification_urls:
        classification_name = class_kvs[classification_id]
        id_name_urls.append((classification_id, classification_name, url))

    return _train_no, sorted(class_names), id_name_urls


# 将图片转为tfrecord
def _convert_tfrecord(_train_no, split_name, id_name_urls, class_names_to_ids, output_dir):
    assert split_name in ['train', 'validation']
    num_per_shard = int(math.ceil(len(id_name_urls) / float(FLAGS.cvt_num_shards)))
    with tf.Graph().as_default():
        image_reader = image_util.ImageReader()
        with tf.Session('') as sess:
          for shard_id in range(FLAGS.cvt_num_shards):
            output_filename = dataset_util.get_dataset_filename(
                _train_no, output_dir, split_name, shard_id, FLAGS.cvt_num_shards)

            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
              start_ndx = shard_id * num_per_shard
              end_ndx = min((shard_id + 1) * num_per_shard, len(id_name_urls))
              for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i+1, len(id_name_urls), shard_id))
                sys.stdout.flush()

                # Read the url:
                class_id, name, url = id_name_urls[i]
                print('\r name:' + name)
                print('\r url:' + url)
                if not os.path.exists(FLAGS.cvt_path_prefix + url):
                    continue
                try:
                    image_data = tf.gfile.FastGFile(FLAGS.cvt_path_prefix + url, 'r').read()
                    height, width = image_reader.read_image_dims(sess, image_data)

                    class_id = class_names_to_ids[name]

                    example = tf_util.image_to_tfexample(
                        image_data, 'jpg', height, width, class_id)
                    tfrecord_writer.write(example.SerializeToString())
                except Exception, e:
                    print("error jpg url:" + url)
                    continue

    mysql_util.update_train_number(_train_no, len(id_name_urls))
    sys.stdout.write('\n')
    sys.stdout.flush()


# def writeTFRecords(index,id_name_urls,broadcastVar,train_no,output_dir,split_name):

#     output_filename = dataset_util._get_dataset_filename(
#             train_no,output_dir, split_name, index)
#     with tf.Graph().as_default():
#         with tf.Session('') as sess:
#             image_reader = image_util.ImageReader()
#             with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
#                 for id,name,url in id_name_urls:
#                     if os.path.exists(_PATH_PRFIX+url):
#                         image_data = tf.gfile.FastGFile(_PATH_PRFIX+url, 'r').read()
#                         height, width = image_reader.read_image_dims(sess, image_data)
#                         class_id = broadcastVar.value[name]
#                         print("the partition %d and url %s height: %d,weight:%d ,image_size:%d" % (index,url,height,width,len(image_data)))
#                         example = tf_util.image_to_tfexample(image_data, 'jpg', height, width, class_id)
#                         tfrecord_writer.write(example.SerializeToString())
#                     else:
#                         print(_PATH_PRFIX+url)
#     return output_filename.encode('utf8')


# def _convert_tfrecord_spark(train_no,split_name, id_name_urls, class_names_to_ids, output_dir,sc):
#     print(len(id_name_urls))
#     url_rdd = sc.parallelize(id_name_urls,_NUM_SHARDS)
#     broadcastVar = sc.broadcast(class_names_to_ids)
#     tmp = url_rdd.mapPartitionsWithIndex(lambda idx,iter: writeTFRecords(idx,iter,broadcastVar,train_no,output_dir,split_name)).collect()
#     print(tmp)


def main(train_no):
    reload(sys)
    sys.setdefaultencoding('utf-8')
    _set_train_no(train_no)
    output_dir = "/data/jg/image/train/tmp/tfrecord"
    train_no, class_names, img_id_name_urls = _get_urls_and_classes(train_no)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    random.seed(FLAGS.cvt_random_seed)
    random.shuffle(img_id_name_urls)

    _num_validation = int(math.ceil(len(img_id_name_urls) * float(FLAGS.cvt_percent_validation) / 100))
    training_name_urls = img_id_name_urls[_num_validation:]
    # validation_name_urls = img_id_name_urls[:_num_validation]
    _convert_tfrecord(train_no, "train", training_name_urls, class_names_to_ids, output_dir)
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_util.write_label_file(train_no, labels_to_class_names, output_dir)


if __name__ == '__main__':
    main("T201706221606037305")
    # sparkconf=SparkConf().setExecutorEnv("LD_LIBRARY_PATH","/usr/lib/jvm/java/jre/lib/amd64/server:/opt/hadoop/lib/native").setExecutorEnv("CLASSPATH","$(/opt/hadoop/bin/hadoop classpath --glob)").setExecutorEnv("PYSPARK_PYTHON","/root/anaconda2/bin/python").setMaster("spark://node17:7077").setAppName("spark convert tf")
    # sc = SparkContext(conf=sparkconf)
    #_convert_tfrecord_spark(train_no,"train",training_name_urls,class_names_to_ids,output_dir,sc)

