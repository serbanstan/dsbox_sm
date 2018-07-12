import numpy as np

from keras import regularizers

from keras.constraints import maxnorm
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

from d3m.metadata.base import PrimitiveMetadata
from d3m.metadata.hyperparams import Uniform

from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

import config as cfg_
import copy

import d3m.container as container
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params

import keras
import tempfile
import typing

Input = container.DataFrame
Output = container.DataFrame

class SM_Params(params.Params):
    model_: typing.Union[Sequential, None]
    inverse_map_: typing.Union[typing.Dict[typing.Any, typing.Any], None]

class SM_Hyperparams(hyperparams.Hyperparams):
    reg_val = Uniform(
        lower = 0,
        upper = 1e-2,
        q = 1e-3,
        default = 1e-4,
        description = 'l2 regularization penalty',
        semantic_types = ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

class SequentialModel(SupervisedLearnerPrimitiveBase[Input, Output, SM_Params, SM_Hyperparams]):
    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "e1c0f608a542504562fbe506c66337f04e293041",
        "version": "1.0.2",
        "name": "SequentialModel",
        "description": "Uses Sequential from Keras to do predictions with previously finely tuned hyperparams.",
        "python_path": "d3m.primitives.dsbox.SequentialModel",
        "original_python_path": "sequential_model.SequentialModel",
        "source": {
            "name": "ISI",
            "contact": "mailto:sstan@usc.edu",
            "uris": [ "https://github.com/serbanstan/dsbox_sm" ]
        },
        "installation": [ cfg_.INSTALLATION ],
        "algorithm_types": ['MULTILAYER_PERCEPTRON'],
        "primitive_family": "CLASSIFICATION",
        "hyperparams_to_tune": ["reg_val"]
    })


    def __init__(self, *, hyperparams : SM_Hyperparams) -> None:
        super().__init__(hyperparams = hyperparams)

    def set_training_data(self, *, inputs : Input, outputs: Output) -> None:
        # initialize the default parameters
        self.validateSplitRate = 0.2
        self.epochs = 30
        self.batchSize = 10

        # work in DF format
        indeM = inputs.shape[1]
                
        self.inputDim = indeM
        self.kindOfcrossEntropy = 'categorical_crossentropy'

        # turn data to ndarray format
        self.training_inputs = inputs.values
        self.training_outputs = to_categorical(self._create_mapping(outputs))
        self.fitted = False


    def fit(self) -> CallResult[None]:
        modelSub = Sequential()
        
        modelSub.add(Dense(100, input_dim = self.inputDim, kernel_regularizer = regularizers.l2(self.hyperparams['reg_val']), activation = 'tanh', kernel_constraint = maxnorm(2)))
        modelSub.add(Dense(self.training_outputs.shape[1], kernel_regularizer = regularizers.l2(self.hyperparams['reg_val']), activation = 'sigmoid'))
        optimizer = Adam(lr = 0.001)
        
        modelSub.compile(loss = self.kindOfcrossEntropy, optimizer = optimizer, metrics = ['accuracy'])

        self.model = modelSub
        self.model.fit(self.training_inputs, self.training_outputs, validation_split = self.validateSplitRate, epochs = self.epochs, batch_size = self.batchSize)

        self.fitted = True

        return CallResult(None, True, self.epochs)
    
    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]:
        prediction = container.DataFrame(self._inverse_mapping(self.model.predict_classes(inputs.values)))
        prediction.index = copy.deepcopy(inputs.index)

        return CallResult(prediction, True, 0)

    def _create_mapping(self, vec):
        # create a mapping from type to float
        self.mapping = dict()
        self.inverse_map = dict()

        res = []

        mapping_index = 0
        for v in vec.values.ravel():
            if v in self.mapping:
                res.append(self.mapping[v])
            else:
                mapping_index = mapping_index + 1

                self.mapping[v] = mapping_index
                self.inverse_map[mapping_index] = v

                res.append(mapping_index)

        return res

    def _inverse_mapping(self, vec):
        return [self.inverse_map[x] for x in vec]

    def get_params(self) -> SM_Params:
        if not self.fitted:
            raise ValueError("Fit not performed")
        return SM_Params(model_ = self.model, inverse_map_ = self.inverse_map)

    def set_params(self, *, params : SM_Params) -> None:
        self.model = params["model_"]
        self.inverse_map = params["inverse_map_"]

    def _annotation(self):
        if self._annotation is not None:
            return self._annotation
        self._annotation = Primitive()
        self._annotation.name = 'SequentialModel'
        self._annotation.task = 'Classification'
        self._annotation.learning_type = 'SupervisedLearning'
        self._annotation.ml_algorithm = ['Keras Sequential']
        self._annotation.tags = ['multilayer_perceptron']
        return self._annotation




    
