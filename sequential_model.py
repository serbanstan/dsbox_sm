import numpy as np

from keras import regularizers

from keras.constraints import maxnorm
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils

from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase 
import d3m.container as container
import d3m.metadata.hyperparams as hyperparams
import d3m.metadata.params as params

import typing

Input = container.DataFrame
Output = container.DataFrame

class SM_Params(params.Params):
    model: typing.Union[Sequential, None]

class SM_Hyperparams(hyperparams.Hyperparams):
    pass

class SequentialModel(SupervisedLearnerPrimitiveBase[Input, Output, SM_Params, SM_Hyperparams]):
    metadata = PrimitiveMetadata({
        "schema": "v0",
        "id": "6cc829ba74786110c636dee4c4ba92bbbd8a8d14",
        "version": "1.0.0",
        "name": "Sequential_Model",
        "description": "Uses Sequential from Keras to do predictions with previously finely tuned hyperparams.",
        "python_path": "d3m.primitives.dsbox.SequentialModel",
        "original_python_path": "sequential_model.SequentialModel",
        "source": {
            "name": "ISI",
            "contact": "mailto:sstan@usc.edu",
            "uris": [ "https://github.com/serbanstan/sequential-model" ]
        },
        "installation": [ cfg_.INSTALLATION ],
        "algorithm_types": [],
        "primitive_family": "",
        "hyperparams_to_tune": []
    })


    def __init__(self, *, hyperparams : SM_Hyperparams) -> None:
        super().__init__(hyperparams = hyperparams)

    def set_training_data(self, *, inputs : Input, outputs: Output) -> None:
        # initialize the default parameters
        self.regVal = 0.0001
        self.actHidden = 'tanh'
        self.actO = 'sigmoid'
        self.maxnormVal = 4
        self.lr = 0.001
        self.validatSplitRate = 0.2
        self.epochs = 30
        self.batchSize = 10

        # work in DF format
        indeM = len(inputs[0])
        numberOfoutlayerUnit = np.unique(outputs).shape[0]
        
        kindOfcrossEntropy = 'categorical_crossentropy'
        
        if numberOfoutlayerUnit == 2:
            numberOfoutlayerUnit = 1
            kindOfcrossEntropy = 'binary_crossentropy'
                
        self.inputDim = indeM
        self.numberOfoutlayerUnit = numberOfoutlayerUnit
        self.kindOfcrossEntropy = kindOfcrossEntropy
        
        self.training_inputs = inputs
        self.training_outputs = np_utils.to_categorical(outputs, num_classes = np.unique(outputs).shape[0])
        self.fitted = False


    def fit(self) -> CallResult[None]:
        modelSub = Sequential()
        
        modelSub.add(Dense(100, input_dim = self.inputDim, kernel_regularizer = regularizers.l2(0.0001), activation = 'tanh', kernel_constraint = maxnorm(2)))
        modelSub.add(Dense(self.numberOfoutlayerUnit, kernel_regularizer = regularizers.l2(0.0001), activation = 'sigmoid'))
        optimizer = Adam(lr = 0.001)
        
        modelSub.compile(loss = self.kindOfcrossEntropy, optimizer = optimizer, metrics = ['accuracy'])
        
        self.model=modelSub
        self.model.fit(self.training_inputs, self.training_outputs,validation_split=self.validatSplitRate,epochs=self.epochs, batch_size=self.batchSize)
        return CallResult(None, True, self.epochs)
    
    def produce(self, *, inputs : Input, timeout : float = None, iterations : int = None) -> CallResult[Output]:
        return CallResult(self.model.predict(inputs), True, 0)

    def get_params(self) -> SM_Params:
        return SM_Params()

    def set_params(self, *, params : SM_Params) -> None:
        self.model = params["model"]
        pass

    
