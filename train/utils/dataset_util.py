# coding=utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

slim = tf.contrib.slim

LABELS_FILENAME = 'labels.txt'

_FILE_PATTERN = '%s_%s_*.tfrecord'

# SPLITS_TO_SIZES = {'train': 50000, 'test': 10000}


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [32 x 32 x 3] color image.',
    'label': 'A single integer between 0 and 9',
}


def write_label_file(train_no, labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.

    Args:
      labels_to_class_names: A map of (integer) labels to class names.
      dataset_dir: The directory in which the labels file should be written.
      filename: The filename where the class names are written.
    """
    label_dir = os.path.join(dataset_dir,train_no)
    if not os.path.exists(label_dir):
      os.makedirs(label_dir)
    labels_filename = os.path.join(label_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
      for label in labels_to_class_names:
        class_name = labels_to_class_names[label]
        f.write('%d:%s\n' % (label, class_name))

def has_labels(train_no, dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir,train_no, filename))


def read_label_file(train_no, dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

    Returns:
        A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir,train_no, filename)
    with tf.gfile.Open(labels_filename, 'r') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
      index = line.index(':')
      labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names


def get_dataset_filename(_train_no, output_dir, split_name, shard_id, num_shard):
    dir = os.path.join(output_dir, _train_no)
    if not os.path.exists(dir):
        os.makedirs(dir)
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
        _train_no, split_name, shard_id, num_shard)
    return os.path.join(dir, output_filename)

def _get_dataset(train_no, split_name, dataset_dir,sample_num, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading cifar10.

    Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

    Returns:
    A `Dataset` namedtuple.

    Raises:
    ValueError: if `split_name` is not a valid train/test split.
    """

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, train_no, file_pattern % (train_no, split_name))

    # Allowing None in the signature so that dataset_factory can use the default.
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

    labels_to_names = None
    if has_labels(train_no, dataset_dir):
        labels_to_names = read_label_file(train_no, dataset_dir)

    return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=sample_num,
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=len(labels_to_names),
      labels_to_names=labels_to_names)
