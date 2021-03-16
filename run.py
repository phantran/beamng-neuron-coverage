#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Phan

import argparse

from src.modules import logger


def get_console_arguments():
    # parse console arguments
    parser = argparse.ArgumentParser(description='Neuron Coverage Demonstration')
    parser.add_argument('--task', dest='task', choices=['collecting', 'data_cleaning', 'training', 'evaluating'],
                        default="",
                        required=True, help='Enter one of the following: collecting, training, evaluating')
    parser.add_argument('--q', dest='number_of_images', type=int, default=1000,
                        required=False, help='Number of images data to collect')
    parser.add_argument('--cm', dest='collect_mode', choices=['manual_mode', 'ai_mode'], default="manual_mode",
                        required=False, help='Data collect mode: manual_mode or ai_mode')
    parser.add_argument('--model', dest='model', choices=['inception_v3', 'cnn'],
                        default="inception_v3",
                        help='Enter one of the following model name for training: inception_v2, cnn')
    parser.add_argument('--mf', dest='model_file', type=str, default="model_1.h5",
                        help='Model file name for running evaluator')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_console_arguments()

    if args.task == "collecting":
        logger.info("Task: Collecting")
        from src.training import data_collector

        data_collector.run_collector(int(args.number_of_images), args.collect_mode)

    if args.task == "data_cleaning":
        logger.info("Task: Cleaning training data")
        from src.training import data_cleaner

        data_cleaner.run_data_cleaner()

    elif args.task == "training":
        logger.info("Task: Training - " + args.model)
        from src.training import model_training

        model_training.run_trainer(args.model)

    elif args.task == "evaluating":
        logger.info("Task: Evaluating")
        from src.evaluation import evaluate

        evaluate.run_evaluator(args.model_file)
