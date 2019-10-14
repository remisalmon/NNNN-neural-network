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

def example_fashion():
    # load data
    from keras.datasets import fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])

    # format data
    train_images = train_images/255 # normalize
    test_images = test_images/255 # normalize

    X_train = train_images.reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[2])).T

    X_test = test_images.reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[2])).T

    Y_train = np.zeros((class_names.shape[0], train_labels.shape[0]))
    for i in range(train_labels.shape[0]):
        Y_train[train_labels[i], i] = 1 # one-hot encode

    Y_test = np.zeros((class_names.shape[0], test_labels.shape[0]))
    for i in range(test_labels.shape[0]):
        Y_test[test_labels[i], i] = 1 # one-hot encode

    # setup NNNN
    nnnn_structure = [
    {'nodes':784, 'activation':None}, # 784 = X.shape[0]
    {'nodes':128, 'activation':nnnn.relu},
    {'nodes':10, 'activation':nnnn.softmax}, # 10 = class_names.shape[0]
    ]
    nnnn_cost = 'CE'

    (w, b) = nnnn.nnnn_init(nnnn_structure)

    # train NNNN
    alpha = 0.01
    iterations = 5

    (w, b, accuracy_hist) = nnnn.nnnn_train(X_train, Y_train, alpha, iterations, w, b, nnnn_structure, nnnn_cost)

    plt.figure()
    plt.plot(accuracy_hist*100)
    plt.xlabel('Training iteration')
    plt.ylabel('Training accuracy (%)')

    # test NNNN
    Y_hat = nnnn.nnnn_test(X_test, w, b, nnnn_structure)

    print('test accuracy = '+str(nnnn.nnnn_accuracy(Y_hat, Y_test)))

    plt.figure()
    for i in np.arange(1, 1+5):
        j = np.random.randint(0, X_test.shape[1])

        plt.subplot(5, 3, 3*i-2)
        plt.imshow(test_images[j, :, :], cmap = 'gray')
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(5, 3, 3*i-1)
        plt.text(0, 0.5, class_names[np.argmax(Y_test[:, j])]+' / Prediction: '+class_names[np.argmax(Y_hat[:, j])]+' ('+str(round(np.max(Y_hat[:, j])*100))+'%)')
        plt.grid(False)
        plt.box(False)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(5, 3, 3*i)
        plt.bar(np.arange(1, 1+len(class_names)), Y_hat[:, j].T)
        plt.xticks(np.arange(1, 1+len(class_names)), ['', '', '', '', '', '', '', '', '', '', ''])
        plt.yticks([])
    plt.xticks(np.arange(1, 1+len(class_names)), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    return

def example_digits():
    # load data
    import sklearn.datasets as sdata
    digits = sdata.load_digits()

    X = digits.data
    C = digits.target

    # format data
    X = (X-X.min())/(X.max()-X.min()) # normalize

    X = X.T
    Y = np.zeros((C.max()+1, C.shape[0]))
    for i in range(C.shape[0]):
        Y[C[i], i] = 1 # one-hot encode

    # select training/testing data subset
    training_samples = 1000
    X_train = X[:, :training_samples]
    Y_train = Y[:, :training_samples]

    X_test = X[:, training_samples:]
    Y_test = Y[:, training_samples:]

    # setup NNNN
    nnnn_structure = [
    {'nodes':64, 'activation':None}, # 64 = X.shape[0]
    {'nodes':30, 'activation':nnnn.sigmoid},
    {'nodes':10, 'activation':nnnn.softmax}, # 10 = Y.shape[0]
    ]
    nnnn_cost = 'CE'

    (w, b) = nnnn.nnnn_init(nnnn_structure)

    # train NNNN
    alpha = 0.01
    iterations = 20

    (w, b, accuracy_hist) = nnnn.nnnn_train(X_train, Y_train, alpha, iterations, w, b, nnnn_structure, nnnn_cost)

    plt.figure()
    plt.plot(accuracy_hist*100)
    plt.xlabel('Training iteration')
    plt.ylabel('Training accuracy (%)')

    # test NNNN
    Y_hat = nnnn.nnnn_test(X_test, w, b, nnnn_structure)

    print('test accuracy = '+str(nnnn.nnnn_accuracy(Y_hat, Y_test)))

    plt.figure()
    for i in np.arange(1, 1+5):
        j = np.random.randint(0, X_test.shape[1])

        plt.subplot(5, 2, 2*i-1)
        plt.imshow(digits.images[training_samples+j], cmap = 'gray')
        plt.axis('off')
        plt.subplot(5, 2, 2*i)
        plt.bar(np.arange(1, 1+10), Y_hat[:, j].T)
        plt.xticks(np.arange(1, 1+10), ['', '', '', '', '', '', '', '', '', '', ''])
        plt.yticks([])
    plt.xticks(np.arange(1, 1+10), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    return

def example_housing():
    # load data
    import sklearn.datasets as sdata
    boston_housing = sdata.load_boston()

    X = boston_housing.data
    Y = boston_housing.target

    # format data
    X = X = (X-X.min(0))/(X.max(0)-X.min(0)) # normalize

    X = X.T
    Y = Y.reshape((1, -1))

    # select training/testing data subset
    training_samples = 406
    X_train = X[:, :training_samples]
    Y_train = Y[:, :training_samples]

    X_test = X[:, training_samples:]
    Y_test = Y[:, training_samples:]

    # setup NNNN
    nnnn_structure = [
    {'nodes':13, 'activation':None}, # 13 = X.shape[0]
    {'nodes':10, 'activation':nnnn.relu},
    {'nodes':10, 'activation':nnnn.relu},
    {'nodes':1, 'activation':nnnn.linear} # 1 = Y.shape[0]
    ]
    nnnn_cost = 'MSE'

    (w, b) = nnnn.nnnn_init(nnnn_structure)

    # train NNNN
    alpha = 0.001
    iterations = 300

    (w, b, accuracy_hist) = nnnn.nnnn_train(X_train, Y_train, alpha, iterations, w, b, nnnn_structure, nnnn_cost)

    plt.figure()
    plt.plot(accuracy_hist*100)
    plt.xlabel('Training iteration')
    plt.ylabel('Training accuracy (%)')
    plt.ylim((0, 100))

    # test NNNN
    Y_hat = nnnn.nnnn_test(X_test, w, b, nnnn_structure)

    print('test accuracy = '+str(nnnn.nnnn_accuracy(Y_hat, Y_test, one_hot = False)))

    print('test MSE = '+str(((Y_hat-Y_test)**2).mean()))

    plt.figure()
    plt.scatter(Y_test, Y_hat)
    plt.xlabel('Testing')
    plt.ylabel('Prediction')

    return

def main():
    example_digits()
    #example_fashion()
    #example_housing()

if __name__ == '__main__':
    main()
