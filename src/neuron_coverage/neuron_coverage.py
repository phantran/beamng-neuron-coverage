import numpy as np

from keras.models import Model

from src.modules import config


class NeuronCoverage:

    def __init__(self, model, threshold=None, excluded_layer=None):
        if excluded_layer is None:
            excluded_layer = ['pool', 'fc', 'flatten', 'input']
        if threshold is None:
            self.threshold = float(config.get("evaluation.neuron_cover_threshold"))
        if model is None:
            raise RuntimeError('Model needs to be a keras model')
        self.model: Model = model
        # the layers that are considered in neuron coverage computation
        self.included_layers = []
        for layer in self.model.layers:
            if all(ex not in layer.name for ex in excluded_layer):
                self.included_layers.append(layer.name)
        # init coverage table
        self.coverage_tracker = {}
        try:
            for layer_name in self.included_layers:
                for index in range(self.model.get_layer(layer_name).output_shape[-1]):
                    self.coverage_tracker[(layer_name, index)] = False
        except Exception as ex:
            raise Exception(f"Error while checking model layer to initialize neuron coverage tracker: {ex}")

    @staticmethod
    def normalize(layer_outputs):
        # Normalize layout output to 0-1 range
        r = (layer_outputs.max() - layer_outputs.min())
        if r == 0:
            return np.zeros(shape=layer_outputs.shape)
        return (layer_outputs - layer_outputs.min()) / r

    def fill_coverage_tracker(self, input_data):
        """
        Given the input, update the neuron covered in the model by this input.
            This includes mark the neurons covered by this input as "covered"
        :param accumulative: find accumulative coverage or not
        :param input_data: the input image
        :return: the neurons that can be covered by the input
        """
        for layer_name in self.included_layers:
            layer_model = Model(self.model.inputs,
                                self.model.get_layer(layer_name).output)

            layer_outputs = layer_model.predict(input_data)
            for layer_output in layer_outputs:
                normalized_val = self.normalize(layer_output)
                for neuron_idx in range(normalized_val.shape[-1]):
                    if np.mean(normalized_val[..., neuron_idx]) > self.threshold:
                        self.coverage_tracker[(layer_name, neuron_idx)] = True
            del layer_outputs
            del layer_model

    def reset_coverage_tracker(self):
        """
        Reset the coverage table
        :return:
        """
        for layer_name in self.included_layers:
            for index in range(self.model.get_layer(layer_name).output_shape[-1]):
                self.coverage_tracker[(layer_name, index)] = False

    def calculate_coverage(self):
        covered_neurons = len([v for v in self.coverage_tracker.values() if v])
        total_neurons = len(self.coverage_tracker)
        return covered_neurons, total_neurons, covered_neurons / float(total_neurons)
