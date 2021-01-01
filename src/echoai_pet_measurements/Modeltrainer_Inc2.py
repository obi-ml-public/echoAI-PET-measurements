import os
import gc
import glob
import pickle
import pandas as pd
import numpy as np
import pdb
from scipy.stats import spearmanr

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.models import load_model

# Custom imports
from echoai_pet_measurements.TFRprovider import DatasetProvider
from echoai_pet_measurements.Inc2 import Inc2model

#%% Custom callbacks for information about training

class Validate(Callback):
    """ Compute correlation coefficient after each epoch """

    def __init__(self, dataset, labels, n_steps):
        super().__init__()
        self.dataset = dataset
        self.labels = labels
        self.n_steps = n_steps

    def on_epoch_end(self, epoch, logs=None):
        print('Evaluation with {} steps from {} samples:'.format(self.n_steps+1, len(self.labels)))
        predictions = list(self.model.predict(self.dataset, verbose=1, steps=self.n_steps + 1))
        sp = spearmanr(self.labels, predictions)
        print('Correlation: {:.3f}'.format(sp[0]))
        print('p-value:     {:.3f}'.format(sp[1]))
        tf.summary.scalar('spearmanr', data=sp[0], step=epoch)

#%% Video trainer

class VideoTrainer:

    def __init__(self, log_dir, model_dict, train_dict, feature_dict):
        self.log_dir = log_dir
        self.model_dict = model_dict
        self.train_dict = train_dict
        self.feature_dict = feature_dict

    def create_dataset_provider(self, augment):
        dataset_provider = DatasetProvider(feature_dict=self.feature_dict,
                                           output_height=self.model_dict['im_size'][0],
                                           output_width=self.model_dict['im_size'][1],
                                           im_scale_factor=self.model_dict['im_scale_factor'],
                                           augment=augment,
                                           model_output=self.model_dict['model_output'])
        return dataset_provider

    def count_steps_per_epoch(self, tfr_file_list, batch_size):
        """ Calculate the number of batches required to run one epoch
            We use the .parquet files to do this
        """
        # We assume, that the .parquet file with all training samples has the same name (except the extension)
        parquet_file_list = [file.replace('.tfrecords', '.parquet') for file in tfr_file_list]
        df = pd.concat([pd.read_parquet(file) for file in parquet_file_list], axis=0, ignore_index=True)
        n_records = len(df.filename.unique())
        steps_per_epoch = int(np.floor(n_records / batch_size)) + 1
        return steps_per_epoch

    def compile_inc2model(self):
        """ Set up the model with loss function, metrics, etc. """

        # Build the model in the distribution strategy scope
        #mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.train_dict['train_device_list'])
        #with mirrored_strategy.scope():
        # Define the model
        model = Inc2model(model_dict=self.model_dict).video_encoder()
        # Loss and accuracy metrics for each output
        loss = {'score_output': tf.keras.losses.MeanSquaredError()}
        # Metrics
        metrics = {'score_output': tf.keras.metrics.MeanAbsolutePercentageError()}
        # Optimizer
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.train_dict['learning_rate'])
        # Compile the model
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metrics)
        return model

    def create_callbacks(self):
        """ Callbacks for model checkpoints and tensorboard visualizations """
        checkpoint_name = self.model_dict['name']+'_chkpt_{epoch:03d}'+'.h5'
        checkpoint_file = os.path.join(self.log_dir, checkpoint_name)

        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_file,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_best_only=False,
                                              save_freq='epoch')

        tensorboard_callback = TensorBoard(log_dir=self.log_dir,
                                           histogram_freq=1,
                                           write_graph=True,
                                           update_freq=10,
                                           profile_batch=0,
                                           embeddings_freq=0)

        callback_list = [checkpoint_callback, tensorboard_callback]

        return callback_list

    def train(self, model, checkpoint_file=None, initial_epoch=0):
        """ Set up the training loop using model.fit """

        # Create datasets
        train_steps_per_epoch = self.count_steps_per_epoch(tfr_file_list=self.train_dict['train_file_list'],
                                                           batch_size=self.train_dict['train_batch_size'])

        trainset_provider = self.create_dataset_provider(augment=self.train_dict['augment'])
        evalset_provider = self.create_dataset_provider(augment=False)

        train_set = trainset_provider.make_batch(tfr_file_list=self.train_dict['train_file_list'],
                                                 batch_size=self.train_dict['train_batch_size'],
                                                 shuffle=True,
                                                 buffer_n_steps=train_steps_per_epoch,
                                                 repeat_count=None,
                                                 drop_remainder=True)

        eval_set = evalset_provider.make_batch(tfr_file_list=self.train_dict['eval_file_list'],
                                               batch_size=self.train_dict['eval_batch_size'],
                                               shuffle=False,
                                               buffer_n_steps=None,
                                               repeat_count=1,
                                               drop_remainder=True)
        # All labels from validation set
        labels = []
        n_steps = 0
        for n_steps, batch in enumerate(eval_set):
            labels.extend(batch[1]['score_output'].numpy())

        callback_list = self.create_callbacks()
        callback_list.append(Validate(eval_set, labels, n_steps))

        # File writer for custom metrics
        file_writer = tf.summary.create_file_writer(self.log_dir+'/metrics')
        file_writer.set_as_default()

        # Load weights from checkpoint
        if checkpoint_file is not None:
            # Loads the weights
            model.load_weights(checkpoint_file)

        hist = model.fit(x=train_set,
                         epochs=self.train_dict['n_epochs'],
                         verbose=self.train_dict['verbose'],
                         validation_data=eval_set,
                         shuffle=True,
                         initial_epoch=initial_epoch,
                         steps_per_epoch=train_steps_per_epoch,
                         validation_steps=self.train_dict['validation_batches'],
                         validation_freq=self.train_dict['validation_freq'],
                         callbacks=callback_list)

        # After fit, save the model and weights
        model.save(os.path.join(self.log_dir, self.model_dict['name'] + '.h5'))

        return hist

    def predict_on_test(self, test_tfr_file_list, checkpoint_file, batch_size):

        # Create test set
        testset_provider = self.create_dataset_provider(augment=False)
        testset = testset_provider.make_batch(tfr_file_list=test_tfr_file_list,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              buffer_n_steps=None,
                                              repeat_count=1,
                                              drop_remainder=False)
        n_steps = 0
        score_list = []
        print('Extracting true labels from testset.')
        for n_steps, batch in enumerate(testset):
            score_list.extend(batch[1]['score_output'].numpy())
        n_steps += 1
        print('Samples: {}, steps: {}'.format(len(score_list), n_steps))

        model = load_model(checkpoint_file)
        predictions = model.predict(testset, verbose=1, steps=n_steps)
        predictions_list = [pred[0] for pred in predictions]

        pred_col_name = os.path.basename(checkpoint_file).split('.')[0]
        label_col_name = self.model_dict['model_output']
        pred_df = pd.DataFrame({label_col_name: score_list,
                                pred_col_name: predictions_list})
        return pred_df








