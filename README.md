# MultiLayeredAdaline
A Back-Propagating Neural Network for Supervised Machine Learning.

Backpropagating networks are considered one of the best off-the-shelf general purpose machine learning methods. This program takes an input vector and puts it thru a series of alternating linear transformations and sigmoid functions to produce an output. The parameters of the linear transformations are adjusted with on-line supervised training using gradient decent to minimize the squared error vector.

The configurations of the network are very flexible. Any number of layers, neurons, or activation functions can be specified.

    ''LayeredNetwork''
    This class represents layers of n neurons and the connections between adjacent layers,
            as a series of n-dimensional vectors and n x m matrices between vector spaces of vectors adjacent in the series.
        Each layer is determined by applying an activation function to each neuron of the previous layer,
            and then applying a linear transformation to the resulting vector.
        Training is conducted by adjusting individual connection weights with back-propagation,
            moving down the mean squared error's gradient relative to the weights.
            Activation functions are also translated horizontally to adjust each neurons firing threshold,
                represented by the connections from an invisible bias node whose activation is always 1.
     
      Attributes:
      ''nInputs'' :: the dimension of network input.
              must be: integer

      ''nOutputs'' :: the dimension of network output.
              must be: integer

      ''nHiddens'' :: the number of nodes in hidden layers, not including bias nodes.
              can be: integer, will make a single hidden layer with that number of nodes.
                  or: list of integers, will make a hidden layer for every integer, from input to output.
                  or: 2-tuple of ints (a, b), will make 'a' hidden layers with 'b' nodes each.

      ''actFunctions'' :: activation function for each layer. All neurons in a given layer have the same activation function.
              can be: list of functions, with length equal to number of layers, including input and output layers,
                  or: 3-tuple of form (fInput, fHidden, fOutput), where 'fInput', 'fOutput' are activation functions,
                      'fHid' can be: function that will be assigned to all hidden layers,
                                 or: list of functions of length nHidden.
              By default, this is linear for input and output, and logistic for hidden layers.

      ''classify'':: if True, the outputs will be stepped to 0 or 1 based on if their activation is greater than .5
              This is false by default.
              
    Methods:
        ''train''  Adjusts weights
            'tolerance' is acceptable mean square error.
            'goal' is acceptable portion classified correctly (classifying only)
            
        ''Evaluate''
            Takes input values in an array or list, and returns output values in an array.
                If inhibit_classify, the returned outputs will not be put thru the step function, even if self._classify==True.

    ''TrainingData'' is a class to store data to be passed to the 'train' method.
      input must be: 3-tuple; (title, inputs, outputs).
                  Where inputs and outputs are 2-D arrays where
                        rows = number of input vectors to evaluate,
                        columns = number of input/output nodes.


Here is a proof that that the backpropagation alogrithm actually finds the error gradient.

![alt text](/BackPropagationProof.jpg)

###variable legend:

n; net value.

A; activation function.

d; desired output. 

e; error.

w; weight.

s; sensitivity.

