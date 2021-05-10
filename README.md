# BEAMNG NEURON COVERAGE

## Introduction

The main purpose of this source code is to demonstrate the correlation between image transformations and neuron
coverage. A model is trained on generated BeamNG dataset for the simple task of keeping vehicles on the road. A part of
the dataset is then transformed (rotate, adjust brightness, etc...) and input to the trained model. Neuron coverage
and output from the trained model on both original and transformed image data will be recorded and summarize to verify
the correlation.

## Setup

Clone <strong>config.yml.sample</strong> file and rename it to <strong>config.yml</strong>. Modify the configurations corresponding to your local setup.

## Running
`python run.py --task TASK_TO_PERFORM` <br>
Replace TASK_TO_PERFORM with one of the following options: <strong>collecting, data_cleaning, training, evaluating</strong>
