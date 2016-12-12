# Benjamin Lloyd Cloer
# trainingdata

from numpy import *
from activationfunctions import *
from layerednetwork import *

__all__ = ['TrainingData']


class TrainingData(object):
    """ Class object to be passed to the 'train' method.
        ''training_data'' the inputs to be trained on and their corresponding desired outputs.
            must be: 3-tuple; (title, inputs, outputs).
        
    """
    def __init__(self, training_data):
        (title, inputs, desires) = training_data
        
        assert isinstance(title, str)
        assert isinstance(inputs, ndarray)
        assert isinstance(desires, ndarray)
        
        nSamples, nInputs = inputs.shape
        mSamples, nOutputs = desires.shape
        
        assert nSamples == mSamples
        
        self.title    = title
        self.inputs   = inputs
        self.desires  = desires
        self.nInputs  = nInputs
        self.nOutputs = nOutputs
        self.nSamples = nSamples

    
