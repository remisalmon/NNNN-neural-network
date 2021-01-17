# Copyright (c) 2021 Remi Salmon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# imports
import numpy as np

# globals
SEED = 888

# functions
def relu(x):
    return (x > 0)*x

def d_relu(x):
    return (x > 0)*1.0

def logistic(x):
    return 1.0/(1.0+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 0)

# main
class NNNN:
    def __init__(self, layers, regression = False):
        """
        Instance a NNNN network

        layers[0] is the input dimension, layers[-1] is the output dimension

        if regression is True, the output is optimized for regression
        """

        self.rg = np.random.default_rng(SEED)

        self.d = len(layers)-1 # depth
        self.W = [] # weight matrices
        self.B = [] # bias matrices
        self.c = not regression # True|False if classification|regression
        self.bc = True if (layers[-1] == 1 and self.c) else False # True|False if binary|multiclass classification

        for i in range(self.d):
            w = self.rg.standard_normal(size = (layers[i+1], layers[i]))*np.sqrt(2.0/layers[i]) # He initialization
            b = np.zeros((layers[i+1], 1))

            self.W.append(w)
            self.B.append(b)

        return

    def __repr__(self):
        r = 'NNNN {}{} network ({} layers)'.format('binary ' if self.bc else '',
                                                   'classification' if self.c else 'regression',
                                                   self.d+1)

        return r

    def predict(self, X, hist = False):
        """
        Predict and return output Y for input X

        X.shape = (n_samples, n_dimensions)
        Y.shape = (n_samples, n_dimensions) or (n_samples,)

        if hist is True, returns output at each layer as a list Y_hist
        if hist is True, Y_hist[i].shape = (n_dimensions, n_samples)
        """

        Y = X.T

        Y_hist = [Y]

        for i in range(self.d-1):
            Y = relu(self.W[i]@Y+self.B[i]) # hidden layers

            Y_hist.append(Y)

        Y = self.W[-1]@Y+self.B[-1] # output layer

        if self.c:
            Y = logistic(Y) if self.bc else softmax(Y) # output layer (classification)

        Y_hist.append(Y)

        return Y_hist if hist else Y.T.squeeze()

    def _grad(self, x, t):
        y_hist = self.predict(x.T, hist = True)

        grad_W = []
        grad_B = []

        for i in range(self.d):
            if i == 0:
                delta = y_hist[-1]-t # d_loss/d_y, loss = 0.5*squared error(y) (regression) or cross-entropy(logistic(y)|softmax(y)) (classification)

            else:
                delta = (self.W[-1-i+1].T@delta)*d_relu(self.W[-1-i]@y_hist[-1-i-1])

            grad_W.append(delta@y_hist[-1-i-1].T)
            grad_B.append(delta)

        y = y_hist[-1]
        grad_W = grad_W[::-1]
        grad_B = grad_B[::-1]

        return y, grad_W, grad_B

    def _loss(self, Y, T):
        if self.c:
            loss = -T*np.log(Y)-(1-T)*np.log(1-Y) if self.bc else -np.sum(T*np.log(Y), axis = 1) # cross-entropy (classification)

        else:
            loss = 0.5*np.sum((Y-T)**2, axis = 1) # squared error (regression)

        loss = np.mean(loss, axis = 0)

        return loss

    def train(self, X, T, iterations, rate = 0.001, alpha = 0.0001):
        """
        Train network with input X and target output T, return loss function history

        X.shape = (n_samples, n_dimensions)
        T.shape = (n_samples, n_dimensions) or (n_samples,)

        iterations is the number of stochastic gradient descent runs
        rate is the learning rate (default: 0.001)
        alpha is the regularization factor (default: 0.0001)
        """

        if T.ndim == 1:
            T = T.reshape((-1, 1))

        Y = np.zeros(T.shape)
        loss_hist = np.zeros(iterations)

        for i in range(iterations):
            for n in self.rg.permutation(len(T)): # stochastic gradient descent
                x = X[n].reshape((-1, 1))
                t = T[n].reshape((-1, 1))

                y, grad_W, grad_B = self._grad(x, t)

                Y[n] = y.reshape(-1)

                for k in range(self.d):
                    self.W[k] -= rate*(grad_W[k]+alpha*self.W[k])
                    self.B[k] -= rate*(grad_B[k]+alpha*self.B[k])

            loss_hist[i] = self._loss(Y, T)+0.5*alpha*sum([np.sum(W**2) for W in self.W]) # data + regularization loss

            print('iteration {}/{} loss = {}'.format(i+1, iterations, loss_hist[i]))

        return loss_hist
