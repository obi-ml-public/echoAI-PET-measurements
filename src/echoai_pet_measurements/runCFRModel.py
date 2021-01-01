"""
Model predictions from checkpoint file
Compile all data as .npy array and expand into memory
"""

#%% Imports

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

from werdich_cfr.tfutils.TFRprovider import DatasetProvider
from werdich_cfr.utils.processing import Videoconverter
from werdich_cfr.tfutils.tfutils import use_gpu_devices

#%% Video selection and model parameters

# GPU parameters
physical_devices, device_list = use_gpu_devices(gpu_device_string='0,1')
batch_size = 1

# Project directory
project_dir = os.path.join(os.environ['HOME'], 'cfrmodel')

# Video selection criteria
max_frame_time_ms = 33.34 # Maximum frame_time acceptable in ms
min_rate = 1/max_frame_time_ms*1e3
min_frames = 40 # Minimum number of frames at min_rate (2 s)
min_length = max_frame_time_ms*min_frames*1e-3

feature_dict = \
    {
    'array': ['image', 'shape'],
    'float': ['rest_global_mbf', 'stress_global_mbf', 'global_cfr_calc'],
    'int': ['record'],
    'features': ['image', 'shape',
    'rest_global_mbf', 'stress_global_mbf',
    'global_cfr_calc', 'record']
    }

model_dict = \
    {
    'name': 'model',
    'im_size': (299, 299, 1),
    'im_scale_factor': 1.177,
    'max_frame_time_ms': 33.34,
    'n_frames': 40,
    'filters': 64,
    'fc_nodes': 1,
    'model_output': 'rest_global_mbf',
    'kernel_init': tf.keras.initializers.GlorotNormal(),
    'bias_init': tf.keras.initializers.Zeros()
    }

#%% Some small helper functions

def get_im_generator(im_array_list):
    """ Yield successive images from list """
    def im_generator():
        for element in im_array_list:
            yield (element[0], element[1])
    return im_generator

def predict_from_array_list(model, model_dict, feature_dict, array_list, batch_size):
    '''
    Predict from list of echo videos
    model: compiled (or loaded from checkpoint) keras model
    model_dict: dict with model hyperparameters
    feature_dict: names of input and output tensors
    array_list: list of tuples with video and shape (array, array.shape)
    '''

    im_generator = get_im_generator(array_list)

    dsp = DatasetProvider(augment=False,
                          im_scale_factor=model_dict['im_scale_factor'],
                          feature_dict=feature_dict)

    dset = tf.data.Dataset.from_generator(generator=im_generator,
                                          output_types=(tf.int32, tf.int32),
                                          output_shapes=(tf.TensorShape([None, None, model_dict['n_frames']]),
                                                         tf.TensorShape([3])))
    dset = dset.map(dsp._process_image)
    dset = dset.map(lambda x: ({'video': x}, {'score_output': 0}))
    dset = dset.batch(batch_size=batch_size, drop_remainder=False).repeat(count=1)

    predict_list = list(np.ndarray.flatten(model.predict(dset, verbose=1, steps=n_steps)))

    return predict_list

#%% Load video data into memory and start preprocessing
video_meta_dict = \
    {
    'filename': ['KX0000D1.npy.lz4'],
    'dir': ['/home/andreas/cfrmodel'],
    'frame_time': [20.702],
    'deltaX': [0.03667],
    'deltaY': [0.03367]
    }

video_meta_df = pd.DataFrame(video_meta_dict)
VC = Videoconverter(max_frame_time_ms=max_frame_time_ms, min_frames=min_frames, meta_df=video_meta_df)
filename = 'KX0000D1.npy.lz4'

image_array_file_list = []
image_array_list = []
meta_disqualified_list = []

error, im = VC.process_video(filename)

if np.any(im):
    image_array_list.append((im, np.asarray(im.shape, np.int32)))
else:
    print('This video file does not qualify.')

#%% Run predictions for all models

# Load model from checkpoint
checkpoint_file = os.path.join(project_dir, 'cfr_a4c_dgx-1_fc1_rest_global_mbf_chkpt_300.h5')
model = load_model(checkpoint_file)
model.summary()

# Predict from image_array_list
n_steps = int(np.ceil(len(image_array_list) / batch_size))
predictions = predict_from_array_list(model=model,
                                      model_dict=model_dict,
                                      feature_dict=feature_dict,
                                      array_list=image_array_list,
                                      batch_size=batch_size)
