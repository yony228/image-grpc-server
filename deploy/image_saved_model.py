# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# !/usr/bin/env python2.7
"""Export inception model given existing training checkpoints.

The model is exported as SavedModel with proper signatures that can be loaded by
standard tensorflow_model_server.
"""

import os.path

import tensorflow as tf
from nets import inception
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

from nets import inception_utils

# This is a placeholder for a Google-internal import.

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/inception_train', """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', '/tmp/inception_output', """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1, """Version number of the model.""")
tf.app.flags.DEFINE_integer('image_size', 299, """Needs to provide same value as in training.""")
tf.app.flags.DEFINE_string('labels_dir', '/tmp', """Directory where to read training labels.""")
FLAGS = tf.app.flags.FLAGS


WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
# SYNSET_FILE = os.path.join(WORKING_DIR, 'imagenet_lsvrc_2015_synsets.txt')
# METADATA_FILE = os.path.join(WORKING_DIR, 'imagenet_metadata.txt')
# LABEL_FILE = os.path.join('/tmp', 'labels.txt')
NUM_CLASSES = 1000
NUM_TOP_CLASSES = 5


def export(step=10):
    with open(os.path.join(FLAGS.labels_dir, 'labels.txt')) as f:
        texts = {}
        global NUM_CLASSES, NUM_TOP_CLASSES
        NUM_CLASSES = 0
        for line in f.read().splitlines():
            parts = line.split(':')
            assert len(parts) == 2
            texts[parts[0]] = parts[1]
            NUM_CLASSES += 1
        NUM_TOP_CLASSES = 5 if NUM_CLASSES >= 5 else NUM_CLASSES

    with tf.Graph().as_default():
        # Build inference model.

        # Input transformation.
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {'image/encoded': tf.FixedLenFeature(shape=[], dtype=tf.string), }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        jpegs = tf_example['image/encoded']
        images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)

        # Run inference.
        with slim.arg_scope(inception_utils.inception_arg_scope()):
            logits, _ = inception.inception_v4(images, num_classes=NUM_CLASSES, is_training=False)
        # Transform output to topK result.
        values, indices = tf.nn.top_k(logits, NUM_TOP_CLASSES)

        # Create a constant string Tensor where the i'th element is
        # the human readable class description for the i'th index.
        class_descriptions = []
        for s in range(0, NUM_CLASSES):
            class_descriptions.append(texts[str(s)])
        print class_descriptions
        class_tensor = tf.constant(class_descriptions)

        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))

        print(FLAGS.checkpoint_dir)
        # os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-5000'),
        init_fn = slim.assign_from_checkpoint_fn(os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-' + str(step)),
                                                 slim.get_model_variables())

        with tf.Session() as sess:
            # Load weights
            init_fn(sess)

            # Export inference model.
            output_path = os.path.join(compat.as_bytes(FLAGS.output_dir), compat.as_bytes(str(FLAGS.model_version)))
            print 'Exporting trained model to', output_path
            builder = saved_model_builder.SavedModelBuilder(output_path)

            # Build the signature_def_map.
            classify_inputs_tensor_info = utils.build_tensor_info(serialized_tf_example)
            classes_output_tensor_info = utils.build_tensor_info(classes)
            scores_output_tensor_info = utils.build_tensor_info(values)

            classification_signature = signature_def_utils.build_signature_def(
                inputs={signature_constants.CLASSIFY_INPUTS: classify_inputs_tensor_info},
                outputs={signature_constants.CLASSIFY_OUTPUT_CLASSES: classes_output_tensor_info,
                         signature_constants.CLASSIFY_OUTPUT_SCORES: scores_output_tensor_info},
                method_name=signature_constants.CLASSIFY_METHOD_NAME
            )

            predict_inputs_tensor_info = utils.build_tensor_info(jpegs)
            prediction_signature = signature_def_utils.build_signature_def(
                inputs={'images': predict_inputs_tensor_info},
                outputs={
                    'classes': classes_output_tensor_info,
                    'scores': scores_output_tensor_info
                },
                method_name=signature_constants.PREDICT_METHOD_NAME
            )

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess,
                [tag_constants.SERVING],
                signature_def_map={
                    'predict_images': prediction_signature,
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,
                },
                legacy_init_op=legacy_init_op
            )

            builder.save()
            print 'Successfully exported model to %s' % FLAGS.output_dir


def preprocess_image(image_buffer):
    """Preprocess JPEG encoded bytes to 3D float Tensor."""

    # Decode the string as an RGB JPEG.
    # Note that the resulting image contains an unknown height and width
    # that is set dynamically by decode_jpeg. In other words, the height
    # and width of image is unknown at compile-time.
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    # After this point, all image pixels reside in [0,1)
    # until the very end, when they're rescaled to (-1, 1).  The various
    # adjust_* ops all require this range for dtype float.
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)
    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(
        image, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
    image = tf.squeeze(image, [0])
    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def main(unused_argv=None):
    export()


def out_def(step_num):
    export(step_num)


if __name__ == '__main__':
    tf.app.run()
