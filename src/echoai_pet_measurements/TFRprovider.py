"""
This file is part of the echoAI-PET-measurements project.
"""

#%% Imports
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

#%% Functions and classes

class DatasetProvider:
    ''' Creates a dataset from a list of .tfrecords files.'''

    def __init__(self,
                 feature_dict=None,
                 class_boundaries=(1.232, 1.556, 2.05),
                 output_height=299,
                 output_width=299,
                 augment=False,
                 im_scale_factor=None,
                 model_output=None):

        if feature_dict is None:
            feature_dict = {'array': ['image', 'shape'],
                            'float': ['cfr', 'rest_mbf', 'stress_mbf'],
                            'int': ['record']}

        self.feature_dict = feature_dict
        self.class_boundaries = class_boundaries
        self.output_height = output_height
        self.output_width = output_width
        self.augment = augment
        self.im_scale_factor = im_scale_factor
        self.model_output = model_output

    @tf.function
    def _class_label(self, cfr_value):
        ''' classification label for cfr value '''
        percentile_list = self.class_boundaries
        label = 0
        if cfr_value < percentile_list[0]:
            label = 0
        elif cfr_value >= percentile_list[-1]:
            label = len(percentile_list)
        for p in range(1, len(percentile_list)):
            if (cfr_value >= percentile_list[p - 1]) & (cfr_value < percentile_list[p]):
                label = p
        return tf.one_hot(label, depth = len(percentile_list)+1)

    def augment_image(self, image):

        # maximum rotation angle in degrees
        max_ang_deg = 30
        max_ang = np.pi / 180 * max_ang_deg

        # Random rotation
        image = tfa.image.rotate(image, tf.random.uniform(shape=[],
                                                          minval=-max_ang, maxval=max_ang,
                                                          dtype=tf.float32),
                                 interpolation='NEAREST')

        # Gaussian noise
        common_type = tf.float32  # Make noise and image of the same type
        gnoise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.25, dtype=common_type)
        image_type_converted = tf.image.convert_image_dtype(image, dtype=common_type, saturate=False)
        image = tf.add(image_type_converted, gnoise)

        # Brightness, contrast
        image = tf.image.random_brightness(image, max_delta=0.5)
        image = tf.image.random_contrast(image, 0.7, 2.5)

        return image

    def _process_image(self, image, shape):

        # original shape is [height, width, frames] -> [frames, height, width, 1]
        # If there is no scale-factor, the images will be resized to fit
        image = tf.reshape(image, shape=shape)
        image = tf.transpose(image, perm=[2, 0, 1])
        image = tf.expand_dims(image, axis=-1)

        if self.im_scale_factor is None:
            image = tf.image.resize_with_pad(image,
                                             target_height=self.output_height,
                                             target_width=self.output_width)
        else:
            # Re-size the image with a single scale factor, then pad or crop to output_size
            im_size = tf.cast(tf.slice(shape, [0], [2]), dtype=tf.float32)
            new_im_size = tf.cast(tf.math.ceil(tf.math.scalar_mul(self.im_scale_factor, im_size)), tf.int32)
            image = tf.image.resize(image, size=new_im_size, antialias=True)
            # Crop or pad to the desired output size
            image = tf.image.resize_with_crop_or_pad(image,
                                                     target_height=self.output_height,
                                                     target_width=self.output_width)

        # Scale image to have mean 0 and variance 1
        image = tf.cast(image, tf.float32)
        image = tf.image.adjust_contrast(image, contrast_factor=5)
        image = tf.image.per_image_standardization(image)

        # Augment images
        if self.augment:
            image = self.augment_image(image)

        return image

    def _parse(self, serialized):

        example = {}

        example_string = {key: tf.io.FixedLenFeature([], tf.string) for key in self.feature_dict['array']}
        example.update(example_string)

        example_float = {key: tf.io.FixedLenFeature([], tf.float32) for key in self.feature_dict['float']}
        example.update(example_float)

        example_int = {key: tf.io.FixedLenFeature([], tf.int64) for key in self.feature_dict['int']}
        example.update(example_int)

        # Extract example from the data record
        example = tf.io.parse_single_example(serialized, example)

        # Convert image to tensor and shape it
        image_raw = tf.io.decode_raw(example['image'], tf.uint16)
        shape = tf.io.decode_raw(example['shape'], tf.uint16)
        shape = tf.cast(shape, tf.int32) # tf.reshape requires int16 or int32 types
        image = tf.reshape(image_raw, shape)

        # Create output tuple

        video_output = {'video': self._process_image(image, shape)}

        # Without a model output parameter, return everything
        if self.model_output is None:
            score_output = {}
            #score_output = {'class_output': self._class_label(example[self.model_output])}
            score_output.update({key: example[key] for key in self.feature_dict['float']})
            score_output.update({key: example[key] for key in self.feature_dict['int']})
        else:
            score_output = {'score_output': example[self.model_output]}
        return (video_output, score_output)

    def make_batch(self, tfr_file_list, batch_size, shuffle,
                   buffer_n_steps=100, repeat_count=1, drop_remainder=False):

        # Shuffle data
        if shuffle:

            #n_parallel_calls = tf.data.experimental.AUTOTUNE

            files = tf.data.Dataset.list_files(tfr_file_list, shuffle=True)

            dataset = files.interleave(tf.data.TFRecordDataset,
                                       cycle_length=len(tfr_file_list),
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)

            dataset = dataset.shuffle(buffer_size=buffer_n_steps * batch_size,
                                      reshuffle_each_iteration=True)

        else:
            dataset = tf.data.TFRecordDataset(tfr_file_list)
            #n_parallel_calls = 1

        # Parse records
        dataset = dataset.map(map_func=self._parse,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE).\
            cache().prefetch(tf.data.experimental.AUTOTUNE)

        # Batch it up
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

        # Prefetch
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset.repeat(count=repeat_count)