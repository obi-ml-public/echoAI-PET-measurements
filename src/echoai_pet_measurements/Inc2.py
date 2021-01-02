"""
This file is part of the echoAI-PET-measurements project.
"""

import os
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform, Zeros

from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (BatchNormalization, Conv3D, MaxPooling3D, Dense,
                                     concatenate, GlobalAveragePooling3D, Dropout, Activation)

#%% Model class

class Inc2model:

    """
    INCEPTION 3D MODEL 2
    Output can be cfr or mbf regression
    """

    def __init__(self, model_dict):

        # NETWORK PARAMETERS AS DICTIONARY
        self.im_size = model_dict['im_size']
        self.n_frames = model_dict['n_frames']
        self.fc_nodes = model_dict['fc_nodes']
        self.filters = model_dict['filters']
        self.kernel_init = model_dict['kernel_init']
        self.bias_init = model_dict['bias_init']

    def inception_module(self, x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5,
                         filters_pool_proj):

        # CONV_1x1
        conv_1x1 = Conv3D(filters_1x1, (1, 1, 1), padding='same', activation='relu', kernel_initializer=self.kernel_init,
                          bias_initializer=self.bias_init)(x)
        #conv_1x1 = BatchNormalization(scale=False, trainable=trainable)(conv_1x1)

        # CONV_3x3
        conv_3x3 = Conv3D(filters_3x3_reduce, (1, 1, 1), padding='same', activation='relu',
                          kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        conv_3x3 = BatchNormalization(scale=False)(conv_3x3)
        conv_3x3 = Conv3D(filters_3x3, (3, 3, 3), padding='same', activation='relu', kernel_initializer=self.kernel_init,
                          bias_initializer=self.bias_init)(conv_3x3)
        #conv_3x3 = BatchNormalization(scale=False, trainable=trainable)(conv_3x3)

        # CONV_5x5
        conv_5x5 = Conv3D(filters_5x5_reduce, (1, 1, 1), padding='same', activation='relu',
                          kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        conv_5x5 = BatchNormalization(scale=False)(conv_5x5)
        conv_5x5 = Conv3D(filters_5x5, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                          kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(conv_5x5)
        conv_5x5 = BatchNormalization(scale=False)(conv_5x5)
        conv_5x5 = Conv3D(filters_5x5, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu',
                          kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(conv_5x5)

        # conv_7x7 = BatchNormalization()(conv_5x5)
        # conv_7x7 = Conv3D(filters_5x5, (3,3,3),strides=(1,1,1), padding='same', activation='relu',kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_7x7)

        # MAXPOOLING
        pool_proj = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
        pool_proj = Conv3D(filters_pool_proj, (1, 1, 1), padding='same', activation='relu',
                           kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(pool_proj)

        output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=4)

        return output

    def video_encoder(self):

        video = layers.Input(shape=(self.n_frames, *self.im_size), name='video')

        x = Conv3D(self.filters, (7, 7, 7), padding='same', strides=(2, 2, 2), activation='relu',
                   kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(video)
        x = BatchNormalization(scale=False)(x)
        x = MaxPooling3D(pool_size=(3, 3, 3), padding='same', strides=(2, 2, 2))(x)
        x = Conv3D(self.filters, (1, 1, 1), padding='same', strides=(1, 1, 1), activation='relu',
                   kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        x = BatchNormalization(scale=False)(x)
        x = Conv3D(self.filters * 3, (3, 3, 3), padding='same', strides=(1, 1, 1), activation='relu',
                   kernel_initializer=self.kernel_init, bias_initializer=self.bias_init)(x)
        x = BatchNormalization(scale=False)(x)

        # INCEPTION 1
        x = self.inception_module(x, filters_1x1=self.filters, filters_3x3_reduce=int(self.filters * 1.5), filters_3x3=self.filters * 4,
                                  filters_5x5_reduce=int(self.filters / 4), filters_5x5=int(self.filters / 2),
                                  filters_pool_proj=int(self.filters / 2))
        x = BatchNormalization(scale=False)(x)

        # INCEPTION 2
        x = self.inception_module(x, filters_1x1=self.filters * 2, filters_3x3_reduce=self.filters * 2, filters_3x3=self.filters * 3,
                                  filters_5x5_reduce=int(self.filters / 2), filters_5x5=self.filters * 3, filters_pool_proj=self.filters)

        x = BatchNormalization(scale=False)(x)

        x = MaxPooling3D(pool_size=(1, 3, 3), padding='same', strides=(2, 2, 2))(x)  # (1,3,3)

        # INCEPTION 3
        x = self.inception_module(x, filters_1x1=self.filters * 3, filters_3x3_reduce=int(self.filters * 1.5),
                             filters_3x3=int(self.filters * 3.25), filters_5x5_reduce=int(self.filters / 4),
                             filters_5x5=int(self.filters * 0.75), filters_pool_proj=self.filters)
        x = BatchNormalization(scale=False)(x)

        # INCEPTION 4
        x = self.inception_module(x, filters_1x1=int(self.filters * 2.5), filters_3x3_reduce=int(self.filters * 1.75),
                             filters_3x3=int(self.filters * 3.5), filters_5x5_reduce=int(self.filters * 0.375),
                             filters_5x5=self.filters, filters_pool_proj=self.filters)
        x = BatchNormalization(scale=False)(x)

        # INCEPTION 5
        x = self.inception_module(x, filters_1x1=self.filters * 2, filters_3x3_reduce=self.filters * 2, filters_3x3=self.filters * 4,
                             filters_5x5_reduce=int(self.filters * 0.375), filters_5x5=self.filters, filters_pool_proj=self.filters)
        x = BatchNormalization(scale=False)(x)

       # INCEPTION 6
        x = self.inception_module(x, filters_1x1=int(self.filters * 1.75), filters_3x3_reduce=int(self.filters * 2.25),
                             filters_3x3=int(self.filters * 4.5), filters_5x5_reduce=int(self.filters / 2), filters_5x5=self.filters,
                             filters_pool_proj=self.filters)
        x = BatchNormalization(scale=False)(x)

        # INCEPTION 7
        x = self.inception_module(x, filters_1x1=self.filters * 4, filters_3x3_reduce=int(self.filters * 2.5), filters_3x3=self.filters * 5,
                             filters_5x5_reduce=int(self.filters / 2), filters_5x5=self.filters * 2,
                             filters_pool_proj=self.filters * 2)
        x = BatchNormalization(scale=False)(x)
        x = MaxPooling3D((1, 3, 3), strides=(2, 2, 2))(x)  # (2,3,3) padding='same'

        # INCEPTION 8
        x = self.inception_module(x, filters_1x1=self.filters * 4, filters_3x3_reduce=int(self.filters * 2.5), filters_3x3=self.filters * 5,
                             filters_5x5_reduce=int(self.filters / 2), filters_5x5=self.filters * 2,
                             filters_pool_proj=self.filters * 2)
        x = BatchNormalization(scale=False)(x)

        # INCEPTION 9
        x = self.inception_module(x, filters_1x1=self.filters * 6, filters_3x3_reduce=self.filters * 3, filters_3x3=self.filters * 6,
                             filters_5x5_reduce=int(self.filters * 0.75), filters_5x5=self.filters * 2,
                             filters_pool_proj=self.filters * 2)
        x = BatchNormalization(scale=False)(x)
        x = MaxPooling3D(pool_size=(1, 3, 3))(x)

        # INCEPTION 10
        x = self.inception_module(x, filters_1x1=self.filters * 6, filters_3x3_reduce=self.filters * 3, filters_3x3=self.filters * 6,
                             filters_5x5_reduce=int(self.filters * 0.75), filters_5x5=self.filters * 2,
                             filters_pool_proj=self.filters * 2)
        x = BatchNormalization(scale=False)(x)

        # MBF OUTPUT
        if self.fc_nodes==1:
            score_output = GlobalAveragePooling3D()(x)
            score_output = Dropout(0.4)(score_output)
            score_output = Dense(self.fc_nodes, activation='relu')(score_output)
            score_output = BatchNormalization(scale=False)(score_output)
            score_output = Dense(1, name='score_output', activation=None)(score_output)
        else:
            score_output = Conv3D(self.fc_nodes, (1, 1, 1), activation=None)(x)
            score_output = BatchNormalization(scale=False)(score_output)
            score_output = Activation('relu')(score_output)
            score_output = Conv3D(1, (1, 1, 1), activation=None)(score_output)
            score_output = GlobalAveragePooling3D(name='score_output')(score_output)
        model = Model(inputs=video, outputs=score_output)
        return model
