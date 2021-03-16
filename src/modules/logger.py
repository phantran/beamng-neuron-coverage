#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Phan

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


def info(message: str):
    logging.info(message)
