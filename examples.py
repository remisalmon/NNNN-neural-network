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

import nnnn

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as sk

def example_digits():
    # load data
    digits = sk.load_digits()

    X = digits.data
    C = digits.target

    # format data
    X = (X-X.min())/(X.max()-X.min()) # normalize
    X = X.T

    Y = np.zeros((C.max()+1, C.shape[0]))
    for i in range(C.shape[0]):
        Y[C[i], i] = 1 # one-hot encode

    # select training data subset
    X_train = X[:, :1000]
    Y_train = Y[:, :1000]

    # set up nnnn
    nnnn_structure = [
    {'layers':64, 'activation':None},
    {'layers':30, 'activation':nnnn.relu},
    {'layers':10, 'activation':nnnn.softmax},
    ]
    nnnn_cost = 'CE'

    (w, b) = nnnn.nnnn_init(nnnn_structure)

    # train nnnn
    alpha = 0.01
    iterations = 100

    (w, b, accuracy_hist) = nnnn.nnnn_train(X_train, Y_train, alpha, iterations, w, b, nnnn_structure, nnnn_cost)

    plt.figure()
    plt.plot(accuracy_hist*100)
    plt.xlabel('Training iteration')
    plt.ylabel('Training accuracy (%)')

    # select testing data subset
    X_test = X[:, 1000:]
    Y_test = Y[:, 1000:]

    # test nnnn
    Y_hat = nnnn.nnnn_test(X_test, w, b, nnnn_structure)

    print('test accuracy = '+str(nnnn.nnnn_accuracy(Y_hat, Y_test)))

    return

def main():
    example_digits()

if __name__ == '__main__':
    main()