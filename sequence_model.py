from keras.models import Sequential
from keras.regularizers import l2

from d3m.primitive_interfaces.base import CallResult   
import d3m.container as container    

Input = container.DataFrame
Output = container.DataFrame

class sequence_model:
    def __init__(self, regVal = 0.0001, actHidden = 'tanh', actO = 'sigmoid', maxnormVal = 4, lr = 0.001):
        self.regVal = regVal
        self.actHidden = actHidden
        self.actO = actO
        self.maxnormVal = maxnormVal
        self.lr = lr
        self.validatSplitRate = 0.2
        self.epochs = 30
        self.batchSize = 10

    def set_training_data(self, *, inputs : Input, outputs: Output) -> None:
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

    
