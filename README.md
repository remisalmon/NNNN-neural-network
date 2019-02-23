# NNNN: a Nth NumPy Neural Network

`nnnn.py` is a fully connected feed-forward neural network with stochastic gradient descent written in Python

## Features

* Activation functions: ReLU (`relu`), Sigmoid (`sigmoid`), Softmax (`softmax`)
* Cost functions: Mean Square Error (`MSE`), Binary Cross-Entropy (`BCE`), Categorical Cross-Entropy (`CE`)
* Optimization algorithm: Stochastic Gradient Descent (truly stochastic)
* Only depends on NumPy
* 143 LOC and 5 KB

## Usage

### Initialization

Example:  
For a 4-layers network with 2 ReLU hidden layers, a Softmax output layer and 2-d inputs and outputs:

```
nnnn_structure = [
{'nodes':2, 'activation':None}, # input layer (no activation)
{'nodes':8, 'activation':relu},
{'nodes':8, 'activation':relu},
{'nodes':2, 'activation':softmax}, # output layer
]

(w, b) = nnnn_init(nnnn_structure)
```

### Training

With input and output data `X` and `Y`, a Categorical Cross-Entropy cost function, a gradient descent rate of `0.01` and `1000` interations of the stochastic gradient descent:

```
nnnn_train(X, Y, alpha = 0.01, iterations = 1000, w, b, nnnn_structure)
```

#### Note

Setting the following activation functions in the output layer automatically default to the following cost functions:  
`sigmoid` → `BCE`  
`softmax` → `CE`  
`relu` → `MSE`

The cost functions `BCE` and `CE` require the training output data `Y` to be encoded as one-hot vectors

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

* Add regularization term
* Implement batch/mini-batch gradient descent
* Vectorize more
