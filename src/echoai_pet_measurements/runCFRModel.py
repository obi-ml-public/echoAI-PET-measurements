"""
This file is part of the echoAI-PET-measurements project.
"""

#%% Imports

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from echoai_pet_measurements.TFRprovider import DatasetProvider
from echoai_pet_measurements.processing import Videoconverter
from echoai_pet_measurements.Modeltrainer_Inc2 import VideoTrainer

#%% Video selection and model parameters

# Project directory
project_dir = os.path.join(os.environ['HOME'], 'data', 'cfrmodel')

# Video selection criteria
max_frame_time_ms = 33.34 # Maximum frame_time acceptable in ms
min_rate = 1/max_frame_time_ms*1e3
min_frames = 40 # Minimum number of frames at min_rate (2 s)
min_length = max_frame_time_ms*min_frames*1e-3

# Model parameters
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

feature_dict = \
    {
    'array': ['image', 'shape'],
    'float': ['rest_global_mbf', 'stress_global_mbf', 'global_cfr_calc'],
    'int': ['record'],
    'features': ['image', 'shape', 'rest_global_mbf', 'stress_global_mbf',
    'global_cfr_calc', 'record']
    }

# Training parameters
train_dict = {'train_device_list': None,
              'learning_rate': 0.0001,
              'augment': False,
              'train_batch_size': 64,
              'eval_batch_size': 32,
              'validation_batches': None,
              'validation_freq': 1,
              'n_epochs': 300,
              'verbose': 1,
              'meta_dir': None,
              'train_file_list': None,
              'eval_file_list': None}

#%% Functions

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

    n_steps = int(np.ceil(len(array_list) / batch_size))

    predict_list = list(np.ndarray.flatten(model.predict(dset, verbose=1, steps=n_steps)))

    return predict_list

def runCFRModel(data_array_list,
                frame_time_ms_list,
                deltaX_list,
                deltaY_list,
                batch_size=1,
                checkpoint_dict=None):

    assert len(data_array_list) == len(frame_time_ms_list) == len(deltaX_list) == len(deltaY_list)

    if checkpoint_dict is None:
        checkpoint_dict = {model_dict['model_output']: None}

    # Process the videos
    image_array_list = []
    qualified_index_list = []
    for data_idx in range(len(data_array_list)):
        VC = Videoconverter(max_frame_time_ms=max_frame_time_ms, min_frames=min_frames, meta_df=None)
        error, im = VC.process_data(data=data_array_list[data_idx],
                                    deltaX=deltaX_list[data_idx],
                                    deltaY=deltaY_list[data_idx],
                                    frame_time=frame_time_ms_list[data_idx])
        if np.any(im):
            image_array_list.append((im, np.asarray(im.shape, np.int32)))
            qualified_index_list.append(data_idx)
        else:
            print(f'Video data for index{data_idx} does not qualify: {error}')

    # Model loop
    output_dict = {}
    for model_output in checkpoint_dict.keys():
        model_dict['model_output'] = model_output

        if checkpoint_dict[model_output] is not None:
            model = load_model(checkpoint_dict[model_output])
        else:
            print(f'No weights provided for {model_output}. Initializing new model.')
            # Initialize a new model (just in case a checkpoint file is missing)
            VT = VideoTrainer(log_dir=None, model_dict=model_dict, train_dict=train_dict, feature_dict=feature_dict)
            model = VT.compile_inc2model()

        output_dict[model_output] = predict_from_array_list(model=model,
                                                            model_dict=model_dict,
                                                            feature_dict=feature_dict,
                                                            array_list=image_array_list,
                                                            batch_size=batch_size)

    return qualified_index_list, output_dict
