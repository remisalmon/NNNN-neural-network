# Copyright (c) 2019 Remi Salmon
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

import numpy as np

def sigmoid(x):
    return(1.0/(1.0+np.exp(-x)))

def d_sigmoid(x):
    f = sigmoid(x)
    return(f*(1.0-f))

def relu(x):
    return((x > 0)*x)

def d_relu(x):
    return((x > 0)*1.0)

def linear(x):
    return(x)

def d_linear(x):
    return(1.0)

def softmax(x):
    x = x-x.max()
    return(np.exp(x)/(np.sum(np.exp(x), axis = 0)))

def d_cost_MSE(y_hat, y):
    return(y_hat-y) # df/dy_hat of f = MSE(y_hat) = (1/2)*np.power(y_hat-y, 2)

def d_costactivation_BCE_sigmoid(y_hat, y):
    return(y_hat-y) # df/dy_hat of f = BCE(sigmoid()) with BCE(y_hat) = -(y*np.log(y_hat)+(1-y)*np.log(1-y_hat)))

def d_costactivation_CE_softmax(y_hat, y):
    return(y_hat-y) # df/dy_hat of f = CE(softmax()) with CE(y_hat) = -sum(y*np.log(y_hat)))

def nnnn_accuracy(Y_hat, Y, one_hot = True): # compute nnnn accuracy
    a = 0
    n = Y.shape[1]

    for i in range(n):
        if one_hot: # classification accuracy
            if np.all((Y_hat[:, i] > 0.5)*1 == Y[:, i]):
                a = a+1
        else: # regression accuracy
            a = a+(1-np.linalg.norm(Y_hat[:, i]-Y[:, i])/np.linalg.norm(Y[:, i]))
    a = a/n

    return(a)

def nnnn_init(nnnn_structure): # initialize nnnn weights and biases gradient matrices
    w = {}
    b = {}

    for i in np.arange(1, len(nnnn_structure)):
        w[i] = np.random.randn(nnnn_structure[i]['nodes'], nnnn_structure[i-1]['nodes'])*np.sqrt(2/(nnnn_structure[i]['nodes']+nnnn_structure[i-1]['nodes']))
        b[i] = np.zeros((nnnn_structure[i]['nodes'], 1))

    return(w, b)

def nnnn_forward(x, w, b, nnnn_structure): # compute nnnn output
    z_hist = {}
    a_hist = {}

    a = x
    a_hist[0] = a

    for i in np.arange(1, len(nnnn_structure)):
        activation = nnnn_structure[i]['activation']

        z = np.dot(w[i], a)+b[i]
        a = activation(z)

        z_hist[i] = z
        a_hist[i] = a

    y = a

    return(y, z_hist, a_hist)

def nnnn_grad(x, y, w, b, nnnn_structure, nnnn_cost): # compute nnnn output + weights and biases gradient matrices
    (y_hat, z_hist, a_hist) = nnnn_forward(x, w, b, nnnn_structure)

    dw = {}
    db = {}

    for i in reversed(np.arange(1, len(nnnn_structure))):
        if i == len(nnnn_structure)-1:
            if nnnn_structure[i]['activation'] == softmax:
                delta = d_costactivation_CE_softmax(y_hat, y)

            elif nnnn_structure[i]['activation'] == sigmoid:
                if nnnn_cost == 'BCE':
                    delta = d_costactivation_BCE_sigmoid(y_hat, y)

                elif nnnn_cost == 'MSE':
                    delta = d_cost_MSE(y_hat, y)*d_sigmoid(z_hist[i])

            elif nnnn_structure[i]['activation'] == relu:
                delta = d_cost_MSE(y_hat, y)*d_relu(z_hist[i])

            elif nnnn_structure[i]['activation'] == linear:
                delta = d_cost_MSE(y_hat, y)*d_linear(z_hist[i])

        else:
            if nnnn_structure[i]['activation'] == sigmoid:
                d_activation = d_sigmoid

            elif nnnn_structure[i]['activation'] == relu:
                d_activation = d_relu

            elif nnnn_structure[i]['activation'] == linear:
                d_activation = d_linear

            delta = np.dot(w[i+1].T, delta)*d_activation(z_hist[i])

        dw[i] = np.dot(delta, a_hist[i-1].T)
        db[i] = delta

    return(y_hat, dw, db)

def nnnn_train(X, Y, alpha, iterations, w, b, nnnn_structure, nnnn_cost = None): # train nnnn with X = [nb_dimensions, nb_samples] Y = [nb_classes, nb_samples], nnnn_structure = [size_layer1, ...]
    Y_hat = np.zeros(Y.shape)

    accuracy_hist = np.zeros(iterations)

    one_hot = True if nnnn_structure[-1]['activation'] in [softmax, sigmoid] else False

    n = Y.shape[1]

    for i in range(iterations):
        for j in np.random.permutation(n):
            x = X[:, j].reshape((-1, 1)) # reshape because NumPy
            y = Y[:, j].reshape((-1, 1)) # reshape because NumPy

            (y_hat, dw, db) = nnnn_grad(x, y, w, b, nnnn_structure, nnnn_cost)

            for k in np.arange(1, len(nnnn_structure)):
                w[k] = w[k]-alpha*dw[k]
                b[k] = b[k]-alpha*db[k]

            Y_hat[:, j] = y_hat.reshape((1, -1)) # reshape because NumPy

        accuracy_hist[i] = nnnn_accuracy(Y_hat, Y, one_hot)

        print('iteration '+str(i+1)+'/'+str(iterations)+', accuracy = '+str(accuracy_hist[i]))

    return(w, b, accuracy_hist)

def nnnn_test(x, w, b, nnnn_structure): # compute nnnn output
    a = x

    for i in np.arange(1, len(nnnn_structure)):
        activation = nnnn_structure[i]['activation']

        z = np.dot(w[i], a)+b[i]
        a = activation(z)

    y = a

    return(y)
