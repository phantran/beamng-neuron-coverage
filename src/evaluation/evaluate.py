#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Phan

from matplotlib import rcParams
import os
import random
import time
import traceback

from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Model

from src import definitions
from src.modules import logger, config
from src.modules.config import root
from src.modules.image_processor import ImageProcessor

import matplotlib.pyplot as plt
import numpy as np
from src.neuron_coverage.neuron_coverage import NeuronCoverage

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
rcParams.update({'figure.autolayout': True})


class Evaluator:

    def __init__(self, model_name="model_1.h5"):
        self.evaluation_images_dir = definitions.ROOT_DIR + config.get("evaluation.evaluation_data_path")
        self.model_path = definitions.ROOT_DIR + config.get("data_collecting.model_path") + model_name
        if not os.path.exists(self.model_path):
            raise Exception("Please create a model first before evaluating transformations on neuron coverage")
        self.model: Model = load_model(self.model_path)

    @staticmethod
    def draw_box_plots(res: dict):
        fig, ax = plt.subplots()
        ax.boxplot(res.values())
        ax.set_xticklabels(res.keys(), rotation=40)
        plt.show()

    @staticmethod
    def draw_bar_chart(res: dict):
        neuron_cov = res.values()
        transformation = res.keys()
        y_pos = np.arange(len(res))
        plt.bar(y_pos, neuron_cov, align='center', alpha=0.5)
        plt.xticks(y_pos, transformation, rotation=75)
        plt.ylabel('Neuron Coverage %')
        plt.title('Transformation')
        plt.show()

    @classmethod
    def get_generators(cls, combine, transformation_name, func, processors, images_dir):
        # return a list of generators
        res = [(transformation_name, cls.create_generator(func, images_dir))]
        if transformation_name == "Original":
            return res
        if combine:
            res = [("", cls.create_generator(None, images_dir)),
                   (transformation_name, cls.create_generator(func, images_dir))]
            while len(res) != 3:
                random_trans = None
                rand_name = transformation_name

                def pred():
                    return random_trans is None \
                           or rand_name == transformation_name \
                           or rand_name == "Original"

                while pred():
                    random_trans = random.choice(processors)
                    rand_name = random_trans[0]
                    rand_func = random_trans[1]
                    if not pred():
                        res.append((rand_name, cls.create_generator(rand_func, images_dir)))
                        break
        return res

    @classmethod
    def create_generator(cls, func, images_dir):
        generator = ImageDataGenerator(rescale=1. / 255, preprocessing_function=func)
        evaluation_data_generator = generator.flow_from_directory(images_dir,
                                                                  target_size=(299, 299),
                                                                  class_mode=None,
                                                                  batch_size=1
                                                                  )
        return evaluation_data_generator

    def compute_neuron_coverage(self, single=True, combine=False):
        """
        This function applies a single transformation on a set of original images each time
        to check whether average of neuron coverage is affected by neuron coverage.
        :return:
        """
        result = {}
        images_dir = definitions.ROOT_DIR + "/evaluation/data/images" if single \
            else definitions.ROOT_DIR + "/evaluation/data/"
        processors = ImageProcessor.get_transformations()
        for item in processors:
            nc = NeuronCoverage(self.model)
            transformation_name, func = item[0], item[1]
            generators = self.get_generators(combine, transformation_name, func, processors, images_dir)

            trans_name = transformation_name
            for gen in generators:
                if combine:
                    trans_name = trans_name + " " + gen[0] if gen[0] != transformation_name \
                        else trans_name + ""
                evaluation_data_generator = gen[1]
                nc.fill_coverage_tracker(evaluation_data_generator)

            covered_neurons, total_neurons, coverage = nc.calculate_coverage()
            logger.info(f"Transformation: {trans_name} --- "
                        f"Covered Neurons: {covered_neurons}, "
                        f"Total Neurons: {total_neurons}, "
                        f"Coverage: {coverage}")
            result[trans_name] = coverage

        self.draw_bar_chart(result)

    def run(self):
        logger.info("------------Neuron coverage when applying a single transformation of a single image------------")
        self.compute_neuron_coverage(single=True, combine=False)
        logger.info("------------Neuron coverage when applying combined transformations of a set of images------------")
        self.compute_neuron_coverage(single=False, combine=True)


def run_evaluator(model_file="model.h5"):
    root.file_config("config.yml")
    try:
        start = time.time()
        Evaluator(model_file).run()
        logger.info(f"--------- Execution time: {time.time() - start} seconds ---------")
    except Exception as ex:
        logger.info(f"Error while evaluating neuron coverage {ex}")
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    run_evaluator()
