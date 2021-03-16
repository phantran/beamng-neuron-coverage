#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Phan

import os

from src import definitions
from src.modules import config, logger
from src.modules.config import root
import pandas as pd


def clean_data_csv():
    """
    One might remove recorded images while images are being collected.
    This function synchronizes data in csv files and images in images folder in such cases.
    """
    root.file_config("config.yml")
    data = pd.read_csv(definitions.ROOT_DIR + config.get("data_collecting.csv_path"))
    img_path = definitions.ROOT_DIR + config.get("data_collecting.data_path")
    count = 0
    for i, row in data.iterrows():
        if not os.path.isfile(img_path + row["image_name"]):
            count = count + 1
            data.drop(index=i, inplace=True)
        elif row["steering"] < -5.0 or row["steering"] > 5:
            count = count + 1
            data.drop(index=i, inplace=True)
    data.to_csv(definitions.ROOT_DIR + config.get("data_collecting.csv_path"), index=False, mode='w', header=True)
    logger.info(f"CSV data cleaning successfully!")


def clean_data_images():
    """
    This function remove images in images folder which do not exist in csv file.
    """
    root.file_config("config.yml")
    data = pd.read_csv(definitions.ROOT_DIR + config.get("data_collecting.csv_path"))
    img_path = definitions.ROOT_DIR + config.get("data_collecting.data_path")
    count = 0
    for file in os.listdir(img_path):
        if data["image_name"].str.contains(file).any():
            pass
        else:
            count = count + 1
            os.remove(img_path + file)
    logger.info(f"Image data cleaning successfully! Removed {count} images")


def run_data_cleaner():
    root.file_config("config.yml")
    clean_data_csv()
    clean_data_images()