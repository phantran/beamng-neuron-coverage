#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Phan

import os
import time
import traceback

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, Camera

from src import definitions
from src.modules import config
from src.modules import logger
from src.modules.config import root
import pandas as pd

import imageio


class DataCollector:

    def __init__(self):
        self.collected_data = {"image_name": [], "steering": []}
        self.scenario = Scenario(config.get("data_collecting.scenario_level"),
                                 config.get("data_collecting.scenario_name"))
        self.vehicle = Vehicle('ego_vehicle', model=config.get("data_collecting.vehicle_model"), licence='PYTHON')
        self.bng = BeamNGpy(config.get("beamNG.host"), int(config.get("beamNG.port")), home=config.get("beamNG.home"))

    def launch_beam_ng(self, mode="manual_mode"):
        # Create an ETK800 with the licence plate 'PYTHON'
        electrics = Electrics()
        # attach to get steering angles
        self.vehicle.attach_sensor('electrics', electrics)
        # attach to get images from camera
        self.vehicle.attach_sensor('front_cam', self.create_camera_sensor())
        # Add it to our scenario at this position and rotation
        # self.scenario.add_vehicle(self.vehicle)
        self.scenario.add_vehicle(self.vehicle, pos=tuple(map(float, config.get("data_collecting.pos").split(","))),
                                  rot_quat=tuple(map(float, config.get("data_collecting.rot_quat").split(","))))
        # Place files defining our scenario for the simulator to read
        self.scenario.make(self.bng)
        # Launch BeamNG.research
        self.bng.open()
        # Load and start our scenario
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        if mode == "ai_mode":
            # Make the vehicle's AI span the map
            self.vehicle.ai_drive_in_lane(True)
            self.vehicle.ai_set_mode('span')

    def save_image_manually(self, cam_name='front_cam'):
        img = self.bng.poll_sensors(self.vehicle)[cam_name]['colour']
        steering = self.bng.poll_sensors(self.vehicle)['electrics']['steering']
        self.save_data(img, steering)

    @staticmethod
    def create_camera_sensor(pos=(-0.3, 2, 1.0), direction=(0, 1, 0), fov=100, res=None):
        # Set up camera sensor
        resolution = res
        if res is None:
            resolution = (
                int(config.get("data_collecting.image_width")),
                int(config.get("data_collecting.image_height")))

        pos = pos
        direction = direction
        fov = fov
        front_camera = Camera(pos, direction, fov, resolution,
                              colour=True, depth=True, annotation=True)
        return front_camera

    def save_data(self, img, steering, i: str = "0"):
        file_name = str(int(time.time())) + i + ".jpg"
        try:
            image_path = definitions.ROOT_DIR + config.get('data_collecting.data_path') + file_name
            imageio.imwrite(image_path, np.asarray(img.convert('RGB')), "jpg")
            self.collected_data["image_name"].append(file_name)
            self.collected_data["steering"].append(steering)
        except Exception as ex:
            logger.info(f"Error while saving data -- {ex}")
            raise Exception

    def collect_data(self, number_of_images: int, mode="manual_mode"):
        self.launch_beam_ng(mode)
        logger.info("Start after 3 seconds...")
        time.sleep(5)
        logger.info(f"Start collecting {config.get('data_collecting.number_of_images')} training images")
        i = 0
        exit_normally = True
        try:
            while i < number_of_images:
                # image is training image and steering is label
                img = self.bng.poll_sensors(self.vehicle)['front_cam']['colour']
                steering = self.bng.poll_sensors(self.vehicle)['electrics']['steering']
                logger.info(f"Saving data {i + 1} ...")
                self.save_data(img, steering, str(i))
                logger.info("Saved data successfully")
                i = i + 1
                time.sleep(int(config.get("data_collecting.sleep")))

        except Exception as ex:
            exit_normally = False
            logger.info(f"Error while collecting data {ex}")
        finally:
            self.bng.close()
            return exit_normally

    def save_csv_data(self):
        logger.info("Start saving csv file......")
        csv_path = definitions.ROOT_DIR + config.get('data_collecting.csv_path')
        df = pd.DataFrame(self.collected_data, columns=['image_name', 'steering'])
        if not os.path.isfile(csv_path):
            df.to_csv(csv_path, index=False, header=True)
        else:  # else it exists so append without writing the header
            df.to_csv(csv_path, index=False, mode='a', header=False)


def run_collector(number_of_images=0, mode="manual_mode"):
    root.file_config("config.yml")
    if not number_of_images:
        number_of_images = int(config.get("data_collecting.number_of_images"))

    collector = DataCollector()
    exit_ok = collector.collect_data(number_of_images, mode)
    # Save csv data
    if exit_ok:
        collector.save_csv_data()
        logger.info("Data collected successfully!")
    else:
        logger.info("Saving data...")
        collector.save_csv_data()
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    run_collector()
