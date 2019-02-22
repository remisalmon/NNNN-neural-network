# NNNN: a Nth NumPy Neural Network

`nnnn.py` is a fully connected feedforward neural network with stochastic gradient descent written in Python

## Features

* Activation functions: ReLU (`relu`), Sigmoid (`sigmoid`), Softax (`softmax`)
* Cost functions: Mean Square Error (`MSE`), Binary Cross-Entropy (`BCE`), Categorical Cross-Entropy (`CE`)
* True Stochastic Gradient Descent
* Only 142 LOC and 5KB
* Only requires NumPy

## Usage

### Initialization

Example:  
For a 4-layers network with ReLU hidden layers activations, Softmax output layer activation and 2-d inputs and outputs:

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

The following activation functions in the output automatically default to the following cost functions:  
`sigmoid` → `BCE`  
`softmax` → `CE`  
`relu` → `MSE`

The cost functions `BCE` and `CE` require the training output data `Y` to be a set of one-hot vectors

To use the `MSE` cost function with a `sigmoid` activation function in the output layer, run:

```
nnnn_train(X, Y, alpha = 0.01, iterations = 1000, w, b, nnnn_structure, cost = 'MSE')
```

### Testing

With input data `X`:

```
Y_hat = nnnn_test(X, w, b, nnnn_structure)
```

## Data Format

* `x, y` are NumPy arrays of dimensions `(d, 1), (c, 1)`
* `X, Y` are NumPy arrays of dimensions `(d, n), (c, n)`
* `d` is the input data dimension, `c` is the number of classes, `n` is the number of training/testing samples
* `y, Y` are training data and `y_hat, Y_hat` are network output data (same dimensions as `y, Y`)

## References

* Backpropagation algorithm derivation in matrix form: https://sudeepraja.github.io/Neural/
* Cross-Entropy Loss functions and derivations: https://peterroelants.github.io/posts/cross-entropy-logistic/, https://peterroelants.github.io/posts/cross-entropy-softmax/

## ToDo

* Implement batch/mini-batch gradient descent
* Vectorize more
