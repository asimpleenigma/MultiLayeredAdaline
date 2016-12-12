# Benjamin Lloyd Cloer


from numpy import *
from activationfunctions import *
from trainingdata import *
from collections import Iterable


__all__ = ['LayeredNetwork']


class LayeredNetwork(object):
    """ Represents layers of n neurons and the connections between adjacent layers,
            as a series of n-dimensional vectors and n x m matrices between vector spaces of vectors adjacent in the series.
        Each layer is determined by applying an activation function to each neuron of the previous layer,
            and then applying a linear transformation to the resulting vector.
        Training is conducted by adjusting individual connection weights with back-propagation,
            moving down the mean squared error's gradient relative to the weights.
            Activation functions are also translated horizontally for to adjust each neurons firing threshold,
                represented by the connections from an invisible bias node whose activation is always 1.
    """
    def __init__(self, nInputs, nOutputs, nHiddens=[], actFunctions=(linear, logistic, linear), classify=False):
        """ ''nInputs'' :: the dimension of network input.
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
            ''classify'':: if True, the outputs will be stepped to 0 or 1 based on if their activation is greater than .5
        """
# put inputs in standard format.
    # 'nHiddens' standard format: integer list.
        if isinstance(nHiddens, list):                          # if input 'nHiddens' is list:
            pass                                                        #  input 'nHiddens' is  in standard format.
        elif isinstance(nHiddens, int):                               # if   input 'nHiddens' is integer:
            nHiddens = [nHiddens]                                       #  input 'nHiddens' is length of only hidden layer.
        elif isinstance(nHiddens, tuple):                           # elif input 'nHiddens' is tuple:
            (a, b) = nHiddens                                           #  input 'nHiddens' is (a, b), where
            assert isinstance(a, int) and isinstance(b, int)            #    a, b are integers, and
            nHiddens = a*[b]                                            #    a = number of hidden layers, b = number of neurons in each hidden layer.
        else:                                                       # else: 
            raise TypeError("Input 'nHiddens' has improper form.")  #     input 'nHiddens' is improper.
        
    # 'actFunctions' standard format: function list
        if isinstance(actFunctions, list):                          # if   input 'actFunctions' is list:
            assert len(actFunctions)==len(nHiddens)+2                   # input 'actFunctions' has length equal to the number of layers.
            pass                                                        # input 'actFunctions' is in standard form.
        elif isinstance(actFunctions, tuple):                       # elif input 'actFunctions' is tuple:
            (fIn, fHid, fOut) = actFunctions                            # input 'actFunctions' is (fIn, fHid, fOut),
            assert callable(fIn) and callable(fOut)                     # fIn, fOut are functions.
        # 'fHid' standard format: function list
            if isinstance(fHid, list):                                  # if   input 'fHid' is list:
                assert len(fHid)==len(nHiddens)                             # input 'fHid' length is number of hidden layers.
                pass                                                        # input 'fHid' is in standard form.
            elif callable(fHid):                                        # elif input 'fHid' is function:
                fHid = len(nHiddens)*[fHid]                                 # input 'fHid' is activation function for all hidden layers.
            else:                                                       # else:
                raise  TypeError("Input 'fHidden' has improper form")       #  input 'actFunctions' is improper.
            actFunctions = [fIn] + fHid + [fOut]                        # combine tuple elements into list.
        else:                                                       # else:
            raise  TypeError("Input 'actFunctions' has improper form")  #  input 'actFunctions' is improper.
        
    # assign static attribute values
        self._nL       = len(nHiddens) + 2                          # integer;              network._nL                 = number of layers.
        self._layers   = [nInputs]+nHiddens+[nOutputs]              # integer list;         network._layers[layer]      = number of neurons.
        self._nIn      = nInputs                                    # integer;              network._nOut               = number of neurons in input  layer.
        self._nOut     = nOutputs                                   # integer;              network._nOut               = number of neurons in output layer.
        self._classify = classify                                   # boolean;              network._classify           = step function is applied to outputs.
        self._aFs      = actFunctions                               # function list;        network._aFs[layer]         = activation function.
        self._dAs      = [derivative(f) for f in self._aFs]         # function list;        network._dAs[layer][neuron] = activation derivative function.
        
    # initialize dynamic attribute values.
        self.history = []                                           # dictionary list;      network.history[epoch] = [learning_rate, avg_MSE[, avg_correct]].
        # initialize input layer.
        self.net = [zeros(self._nIn)]                               # 1-D float array list; network.net[layer][neuron]      = net value.
        self.act = [self._aFs[0](self.net[0])]                      # 1-D float array list; network.act[layer][neuron]      = activation value.
        self.w   = []                                               # 2-D float array list; network.w[layer][input][output] = connnection weight.
        # initialize post layers.
        for i in range(self._nL-1):                                 # for each post layer:
            i = i+1                                                     # start count at layer 1.
            self.net += [zeros(self._layers[i])]                        # initialize net values to zero.
            self.act += [self._aFs[i](self.net[i])]                     # initialize activation values.
            self.w   += [random.random((self._layers[i-1]+1, self._layers[i]))]# initialize weights randomly, +1 for bias row of wieghts.



    def evaluate(self, sample, inhibit_classify=False):
        """ Takes input values in an array or list, and returns output values in an array.
                If inhibit_classify, the returned outputs will not be put thru the step function, even if self._classify==True.
        """
        # assign input layer.
        self.net[0] = array(sample)                  # set input layer net values.
        self.act[0] = self._aFs[0](self.net[0])      # activate input layer.

        # calculate post layers.
        for i in range(self._nL-1):                  # for each post layer: 
            i = i+1                                      # integer; i = post layer index.
            pre_act = append(self.act[i-1], 1.)          # 1-D array; pre_act = pre layer activation values, plus bias.
            self.net[i] = dot(pre_act, self.w[i-1])      # post layer net values = pre_act *dot* weights.
            self.act[i] = self._aFs[i](self.net[i])      # set post layer activation values.

        # specify output.
        output = self.act[-1]                        # 1-D float array; output[neuron] = activation value
        if self._classify and not inhibit_classify:  # if self is classifying and uninhibited:
            output = array(map(step, output))            # output is output stepped to 0 or 1.
        return output                                # return (stepped) activation values of output layer.
#Note to self: look into using __cmp__ method for thresholds.
    
    def sensitivities(self, sen_1):
        """ Takes sensitivities of output layer as a 1-D float array, and returns sensitivities of pre layers as 1-D float array list.
        """
        assert len(sen_1) == self._nOut          # input 'sen_1' length is the number of output nodes.
        sen = [sen_1]                            # 1-D float array list; sen[layer][neuron] = sensitivity; d(MSE)/d(self.net[layer][neuron]).
        for i in range(self._nL-1):              # for each post layer:
            i = -1-i                                 # integer;         i = post layer index, counting from final hidden to input.
            pre_dFunc = self._dAs[i-1]               # function;        dAf                 = pre layer activation function derivative.
            pre_dAct  = pre_dFunc(self.net[i-1])     # 1-D float array; dAn[neuron]         = pre layer activation value derivatives.
            wght      = self.w[i][:-1].transpose()   # 2-D float array; wght[output][input] = connection weights, no bias row, transposed.
            pre_s     = pre_dAct * dot(sen[i], wght) # 1-D float array; s[neuron]           = pre layer neuron sensitivity.
            sen       = [pre_s] + sen                # prepend pre layer sensitivity to 'sen'.
        return sen                               # return 'sen'.
    
    def evalStats(self, sample, desire):
        """ Takes two arrays, returns a dictionary of 'actual', 'error', 'MSE', 'sen'.
                If self._classify==True, 'classification', 'correct' are also in the dictionary.
        """
        # calculate error of network for sample.
          # find discrete error.
        if self._classify:                                   # if self is classifying network:
            classification = self.evaluate(sample)               # boolean; classification     = actual classification of sample.
            correct = (classification == desire)                 # boolean; correct            = network classified sample correctly.

          # find continuous error.
        actual = self.evaluate(sample, inhibit_classify=True)# 1-D float array; actual[neuron] = continuous output activation value.
        error  = desire - actual                             # 1-D float array;  error[neuron] = continuous output activation error.
        MSE    = average(error*error)                        # integer;    MSE                 = mean squared error.
        # Note: Executing 'evaluate' above updated self.net, self.act for sample.
        
        # find sensitivities of pre layers using sensitivities of output layer.
        sen = self.sensitivities((-2./self._nOut)*error)     # 1-D float array; sen[layer][neuron] = sensitivity = d(MSE)/d(self.net[layer][neuron]).
        return locals()# statistics                                                  # return 'statistics'.
    
    
    def train(self, training_data, nEpochs=1, learning_rate=.1,
               tolerance=0., goal=1.): # Note to self: add meta learning rate
        """ Adjusts weights to self.w, returns nothing
            Takes 'samples' and 'desired' as arrays where:
                rows = number of input vectors to evaluate,
                columns = number of input/output nodes.
            'tolerance' is acceptable MSE.
            'goal' is acceptable portion correct (classifying only)
        """
        samples = training_data.inputs            # 2-d float array; samples[sample][neuron] = input value.
        desires = training_data.desires           # 2-d float array; desires[sample][neuron] = correct output value.
        nSamples = len(samples)                   # integer;         nSamples                = number of samples to be trained on.
        epoch = 0                                 # integer;         epoch                   = number of epochs trained.
        while epoch < nEpochs:                    # for each epoch:
            epoch += 1                                # track epoch.
            sum_MSE = 0                               # float; sum_MSE sum of mean square errors of samples tested in epoch.
            if self._classify == True:                # if classifying network.
                num_correct = 0                           # integer;         num_correct     = number of samples classified correctly in epoch.
            for h in range(nSamples):                 # for each sample:
                sample = samples[h]                       # 1-D float array; sample[neuron]  = input value.
                desire = desires[h]                       # 1-D float array; desire[neuron]  = desired output value.
                
            # find error, sensitivity
                stats  = self.evalStats(sample, desire)   # dictionary;      stats[stat]     = proformance statistics for sample: actual, error, MSE, sen(, classification, correct).
            # add proformance to epoch average
                sum_MSE += stats['MSE']                   # add MSE of sample to sum_MSE of epoch.
                if self._classify:                        # if classifying network:
                    num_correct += stats['correct']           # if sample was classified correctly, +1 to num_correct.
            # adjust weights
                for i in range(self._nL-1):               # for each pre layer: update weight matrix.
                    activation = append(self.act[i], 1)     # 2-D array; activation = values of pre layer, plus bias node. 
                    self.w[i] += -learning_rate * array( mat(activation).transpose() * mat(stats['sen'][i+1]) ) # adjust weights.

        # update proformance history with epoch stats.
            avg_MSE = sum_MSE/nSamples                # float;       avg_MSE           = epoch average of mean square errors for all samples.
            epoch_stats = [learning_rate, avg_MSE]    # float list;  epoch_stats[stat] = stat value.
            if self._classify == True:                # if classifying network:
                avg_correct = float(num_correct)/nSamples # float;   avg_correct       = the fraction of samples correctly classified in epoch.
                epoch_stats += [avg_correct]              # append avg_correct in epoch_stats.
            self.history += [epoch_stats]             # append epoch_stats to self.history.
        # check if error is tolerable.
            if avg_MSE > tolerance:                # if error is not tolerable:
                continue                                 # continue training.
            if self._classify == True and avg_correct < goal:
                continue
            break


    def hisum(self, nIntervals=10):
        length = len(self.history)
        for epoch in range(0, length, length/nIntervals):
            print self.history[epoch]








                    
                    
                    
