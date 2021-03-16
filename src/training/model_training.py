import time
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3

from keras.layers import Dense, Flatten, Dropout, Input
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D

from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow.python.keras.callbacks import EarlyStopping

from src import definitions
from src.modules import config, logger
from src.modules.config import root


class EpochModel:

    def __init__(self, model="inception_v3"):
        self.save_model_dir = definitions.ROOT_DIR + '/training/models/'
        self.training_images_dir = definitions.ROOT_DIR + config.get("data_collecting.data_path")
        if model == "inception_v3":
            self.epoch_model = self.build_inception_v3()
        elif model == "cnn":
            self.epoch_model = self.build_cnn()
        else:
            raise Exception("Model is not supported")

    @staticmethod
    def build_cnn(image_size=None, weights_path=None):
        image_size = image_size or (299, 299)
        if K.image_data_format() == 'channels_first':
            input_shape = (3,) + image_size
        else:
            input_shape = image_size + (3,)

        img_input = Input(input_shape)

        x = Convolution2D(32, 3, 3, activation='relu', padding='same')(img_input)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(64, 3, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(128, 3, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(0.5)(x)

        y = Flatten()(x)
        y = Dense(1024, activation='relu')(y)
        y = Dropout(.5)(y)
        y = Dense(1)(y)

        model = Model(img_input, y)
        model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_logarithmic_error', metrics=['mse', 'mae'])

        if weights_path:
            model.load_weights(weights_path)

        return model

    @staticmethod
    def build_inception_v3(image_size=None):
        image_size = image_size or (299, 299)
        if K.image_data_format() == 'channels_first':
            input_shape = (3,) + image_size
        else:
            input_shape = image_size + (3,)
        bottleneck_model = InceptionV3(weights='imagenet', include_top=False,
                                       input_tensor=Input(input_shape))
        for layer in bottleneck_model.layers:
            layer.trainable = False

        x = bottleneck_model.input
        y = bottleneck_model.output
        y = GlobalAveragePooling2D()(x)
        y = Dense(1024, activation='relu')(y)
        y = Dropout(.5)(y)
        y = Dense(1)(y)
        model = Model(x, y)
        model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_logarithmic_error', metrics=['mse', 'mae'])
        return model

    @staticmethod
    def get_model_name(k):
        return 'model_' + str(k) + '.h5'

    def train_model(self, training_data_path):
        validation_accuracy = []
        validation_loss = []
        train_data = pd.read_csv(training_data_path)
        train_data[['steering']] = train_data[['steering']].astype(float)
        labels = train_data[['steering']]

        kf = KFold(n_splits=10, random_state=7, shuffle=True)
        generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.3, rescale=1. / 255)

        fold = 1
        for train_index, val_index in kf.split(np.zeros(labels.size), labels):
            training_data = train_data.iloc[train_index]
            validation_data = train_data.iloc[val_index]

            train_data_generator = generator.flow_from_dataframe(training_data, directory=self.training_images_dir,
                                                                 x_col="image_name", y_col="steering",
                                                                 class_mode="raw", shuffle=True)
            validation_data_generator = generator.flow_from_dataframe(validation_data,
                                                                      directory=self.training_images_dir,
                                                                      x_col="image_name", y_col="steering",
                                                                      class_mode="raw", shuffle=True)

            callback1 = tf.keras.callbacks.ModelCheckpoint(self.save_model_dir + self.get_model_name(fold),
                                                           monitor='val_loss', verbose=1,
                                                           save_best_only=True, mode='max')
            callback2 = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
            callbacks_list = [callback1, callback2]

            self.epoch_model.fit(train_data_generator,
                                 epochs=int(config.get("training.epochs")),
                                 callbacks=callbacks_list,
                                 validation_data=validation_data_generator)

            self.epoch_model.load_weights(self.save_model_dir + "model_" + str(fold) + ".h5")

            results = self.epoch_model.evaluate(validation_data_generator)
            results = dict(zip(self.epoch_model.metrics_names, results))
            validation_loss.append(results['loss'])

            tf.keras.backend.clear_session()
            fold += 1
            logger.info("Loss" + validation_loss.__str__())


def run_trainer(model="inception_v3"):
    root.file_config("config.yml")
    try:
        start_time = time.time()
        em = EpochModel(model)
        em.train_model(definitions.ROOT_DIR + config.get("data_collecting.csv_path"))
        logger.info(f"--------- Execution time: {time.time() - start_time} seconds ---------")
    except Exception as ex:
        logger.info(f"Error while training model - {ex}")
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    run_trainer()
