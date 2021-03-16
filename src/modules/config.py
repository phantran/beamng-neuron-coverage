#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Phan

import yaml

from .. import definitions


class Config:

    __config = {}

    def __init__(self):
        self.__config = {}

    def file_config(self, filename: str):
        """Load configuration from file"""
        with open(self.root_path() + filename, 'rt', encoding="utf-8") as ymlfile:
            self.__config = yaml.safe_load(ymlfile)

    @staticmethod
    def root_path():
        return definitions.ROOT_DIR + '/'

    def __setattr__(self, key, value):
        if hasattr(self, key):
            self.__dict__[key] = value
        else:
            self.__config[key] = value

    def get(self, name: str, default=None):
        """Get stored variable from configuration dictionary by name.
        Also you can use dots (.) in variable's name for deeper searching"""
        if name in self.__config:
            return self.__config[name]
        if '.' in name:
            names = name.split('.')
            cur = self.__config
            for name in names:
                if type(cur) is dict and name in cur:
                    cur = cur[name]
                else:
                    return default
            return cur
        return default

    def get_directory_path(self, name: str, default: str, absolute=False):
        """Get stored directory name or default and return absolute path for it"""
        conf_dir = self.get(name, default)
        loc_dir = conf_dir.strip('./') + '/'
        if absolute:
            return self.root_path() + loc_dir
        return loc_dir


root = Config()


def get(name: str, default=None):
    return root.get(name, default)


def get_directory_path(name: str, default: str):
    return root.get_directory_path(name, default, False)


def get_absolute_directory_path(name: str, default: str):
    return root.get_directory_path(name, default, True)

