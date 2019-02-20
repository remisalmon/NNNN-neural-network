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

def f_activation(x): # activation function (sigmoid)
    f = 1.0/(1.0+np.exp(-x))

    return(f)

def df_activation(x): # activation function derivate (sigmoid)
    f = f_activation(x)
    df = f*(1.0-f)

    return(df)

def df_cost(y_hat, y): # cost function derivate (MSE of Binary Cross-Entropy)
    #df = (y_hat-y) # Mean Square Error f = (1/n)*sum((1/2)*np.power(Y_hat-Y, 2))

    df = -(y/y_hat-(1-y)/(1-y_hat)) # Binary Cross-Entropy f = (1/n)*sum(-(Y*np.log(Y_hat)+(1-Y)*np.log(1-Y_hat)))

    return(df)

def f_accuracy(Y_hat, Y): # compute nnnn accuracy
    a = 0
    n = Y.shape[1]

    for i in range(n):
        if np.all((Y_hat[:, i] > 0.5)*1 == Y[:, i]):
                a = a+1
    a = a/n

    return(a)

def nnnn_init(nnnn_structure): # initialize nnnn weights and biases gradient matrices
    w = {}
    b = {}

    for i in np.arange(1, len(nnnn_structure)):
        w[i] = np.random.randn(nnnn_structure[i], nnnn_structure[i-1])
        b[i] = np.random.randn(nnnn_structure[i], 1)

    return(w, b)

def nnnn_forward(x, w, b, nnnn_structure): # compute nnnn output
    a = x

    z_hist = {}
    a_hist = {}

    a_hist[0] = a

    for i in np.arange(1, len(nnnn_structure)):
        z = np.dot(w[i], a)+b[i]
        a = f_activation(z)

        z_hist[i] = z
        a_hist[i] = a

    y = a

    return(y, z_hist, a_hist)

def nnnn_grad(x, y, w, b, nnnn_structure): # compute nnnn output + weights and biases gradient matrices
    (y_hat, z_hist, a_hist) = nnnn_forward(x, w, b, nnnn_structure)

    delta = {}
    dw = {}
    db = {}

    for i in reversed(np.arange(1, len(nnnn_structure))):
        if i == len(nnnn_structure)-1:
            delta[i] = df_cost(y_hat, y)*df_activation(z_hist[i])
        else:
            delta[i] = np.dot(w[i+1].T, delta[i+1])*df_activation(z_hist[i])

        dw[i] = np.dot(delta[i], a_hist[i-1].T)
        db[i] = delta[i]

    return(y_hat, dw, db)

def nnnn_train(X, Y, alpha, iterations, w, b, nnnn_structure): # train nnnn with X = [nb_dimensions, nb_samples] Y = [nb_classes, nb_samples], nnnn_structure = [size_layer1, ...]
    Y_hat = np.zeros(Y.shape)

    accuracy_hist = np.zeros(iterations)

    n = Y.shape[1]

    for i in range(iterations):
        for j in np.random.permutation(n):
            x = X[:, j].reshape((-1, 1)) # because NumPy
            y = Y[:, j].reshape((-1, 1)) # because NumPy

            (y_hat, dw, db) = nnnn_grad(x, y, w, b, nnnn_structure)

            for k in np.arange(1, len(nnnn_structure)):
                w[k] = w[k]-alpha*dw[k]
                b[k] = b[k]-alpha*db[k]

            Y_hat[:, j] = y_hat.reshape((1, -1)) # because NumPy

        accuracy_hist[i] = f_accuracy(Y_hat, Y)

        print('iter '+'%03d'%(i+1)+'/'+str(iterations)+', accuracy = '+str(accuracy_hist[i]))

    return(w, b, accuracy_hist)
