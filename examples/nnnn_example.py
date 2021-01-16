# imports
import numpy as np
import matplotlib.pyplot as plt

from nnnn import NNNN
from sklearn.datasets import load_digits

# globals
TRAIN_TEST_RATIO = 0.5
ITERATIONS = 100
RATE = 0.001
ALPHA = 0.0001

REGRESSION = False

# main
digits = load_digits()

X = digits.data
T = digits.target

X = -1.0+2.0*(X-X.min())/(X.max()-X.min()) # normalize to [-1, 1]

n_train = int(len(X)*TRAIN_TEST_RATIO)

X_train = X[:n_train]
X_test = X[n_train:]

T_train = T[:n_train]
T_test = T[n_train:]

T_train_encoded = np.zeros((n_train, 10))
T_train_encoded[np.arange(n_train), T_train] = 1.0

if not REGRESSION:
    nnnn = NNNN(layers = [64, 16, 10], regression = False)

    loss_hist = nnnn.train(X_train, T_train_encoded, RATE, ALPHA, ITERATIONS)

    fig, ax = plt.subplots()
    ax.plot(loss_hist)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')

    Y_train = nnnn.predict(X_train)
    Y_train_decoded = np.argmax(Y_train, axis = 1)

    Y_test = nnnn.predict(X_test)
    Y_test_decoded = np.argmax(Y_test, axis = 1)

    print('training accuracy = {}%'.format(int(100*sum(Y_train_decoded == T_train)/len(T_train))))
    print('prediction accuracy = {}%'.format(int(100*sum(Y_test_decoded == T_test)/len(T_test))))

    fig, ax = plt.subplots(10, 2)
    for i in range(len(ax)):
        ax[i, 0].imshow(digits.images[n_train+i], cmap = 'gray')
        ax[i, 0].axis('off')
        ax[i, 1].bar(np.arange(10), Y_test[i])
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
    ax[i, 1].set_xticks(np.arange(10))
    ax[0, 0].set_title('target')
    ax[0, 1].set_title('prediction')

if REGRESSION:
    nnnn = NNNN(layers = [64, 32, 16, 8, 1], regression = True)

    loss_hist = nnnn.train(X_train, T_train, RATE, ALPHA, ITERATIONS)

    fig, ax = plt.subplots()
    ax.plot(loss_hist)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')

    Y_train = nnnn.predict(X_train)
    Y_train_decoded = np.round(Y_train).astype(int)

    Y_test = nnnn.predict(X_test)
    Y_test_decoded = np.round(Y_test).astype(int)

    print('training accuracy = {}%'.format(int(100*sum(Y_train_decoded == T_train)/len(T_train))))
    print('prediction accuracy = {}%'.format(int(100*sum(Y_test_decoded == T_test)/len(T_test))))

    fig, ax = plt.subplots(10, 2)
    for i in range(len(ax)):
        ax[i, 0].imshow(digits.images[n_train+i], cmap = 'gray')
        ax[i, 0].axis('off')
        ax[i, 1].text(0.5, 0.5, '{}'.format(Y_test_decoded[i]))
        ax[i, 1].axis('off')
    ax[0, 0].set_title('target')
    ax[0, 1].set_title('prediction')
