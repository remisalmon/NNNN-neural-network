# NNNN: a Nth NumPy Neural Network

`nnnn.py` is a feed-forward neural network with stochastic gradient descent written in Python

## Features

* Activation functions: ReLU (`relu`), Sigmoid (`sigmoid`), Softmax (`softmax`)
* Cost functions: Mean Square Error (`MSE`), Binary Cross-Entropy (`BCE`), Categorical Cross-Entropy (`CE`)
* True Stochastic Gradient Descent
* 142 LOC, 5KB, only requires NumPy

## Usage

### Initialization

Example:  
For a 4-layers network with ReLU hidden layers activations, softmax output layer activation and 2-d inputs and outputs:
```
nnnn_structure = [
{'layers':2, 'activation':None}, # input layer (no activation)
{'layers':8, 'activation':relu},
{'layers':8, 'activation':relu},
{'layers':2, 'activation':softmax}, # output layer
]

(w, b) = nnnn_init(nnnn_structure)
```

### Training

With input and output data `X` and `Y`, a Categorical Cross-Entropy cost function, a gradient descent rate of `0.01` and `1000` interations of the stochastic gradient descent:
```
nnnn_train(X, Y, alpha = 0.01, iterations = 1000, w, b, nnnn_structure)
```

#### Note

In the output layer, the following activation functions automatically default to the following cost functions:  
`sigmoid` → `BCE` 
`softmax` → `CE`  
`relu` → `MSE`

To use `MSE` with a `sigmoid` in the output layer, run:  
```
nnnn_train(X, Y, alpha = 0.01, iterations = 1000, w, b, nnnn_structure, cost = 'MSE')
```

### Testing

With input data `X`:

```
Y_hat = nnnn_test(X, w, b, nnnn_structure)
```

## Data formatting

* `x, y` are vectors of dimensions `(d, 1), (c, 1)`
* `X, Y` are matrices of dimensions `(d, n), (c, n)`
* `d` is the input data dimension, `c` is the number of classes, `n` is the number of training/testing samples
* `y, Y` are training data and `y_hat, Y_hat` are network output data (same dimensions as `y, Y`)

## Comments

Remarks:
* For the stochastic gradient descent, the sum in the cost function `f` is removed since only `y_hat, y` are evaluated instead of the whole training sample `Y_hat, Y` (default algorithm, no need to edit the code)
* To use the batch gradient descent, multiply `df` by `(1/n)`, add up all `dw, db` before updating `w, b` and replace `np.random.permutation(n)` by `range(n)` (need to edit the code)

References:
* Backpropagation algorithm derivation in matrix form: https://sudeepraja.github.io/Neural/
* Cross-Entropy Loss functions and derivations: https://peterroelants.github.io/posts/cross-entropy-logistic/, https://peterroelants.github.io/posts/cross-entropy-softmax/

## ToDo

* Implement batch/mini-batch gradient descent
* Vectorize more
