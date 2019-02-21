# NNNN: a Nth NumPy Neural Network

`nnnn.py` is a feed-forward neural network with stochastic gradient descent written in Python

`nnnn.py` is also:  
* Simple (125 lines of code)  
* Customizable (available activation functions: ReLU, sigmoid; available cost functions: MSE, binary cross-entropy)  
* Vectorized (not completely, the stochastic gradient descent can be vectorized...)

## Usage

### Initialization

For a 4-layers network with ReLU hidden layers activation, sigmoid output layer activation and 2-d input and output:
```
nnnn_structure = [
{'layers':2, 'activation':None}, # input layer (no activation)
{'layers':8, 'activation':relu},
{'layers':8, 'activation':relu},
{'layers':8, 'activation':relu},
{'layers':2, 'activation':sigmoid}, # output layer
]

(w, b) = nnnn_init(nnnn_structure)
```
### Training

With input data `X` and output data `Y`, a gradient descent rate of `0.01` and `1000` interations of the gradient descent:

```
nnnn_train(X, Y, alpha = 0.01, iterations = 1000, w, b, nnnn_structure)
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
* In `df_cost()`:  
For the stochastic gradient descent, the sum in the cost function `f` is removed since only `y_hat, y` are evaluated instead of the whole training sample `Y_hat, Y` (default algorithm, no need to edit the code)  
To use the batch gradient descent, multiply `df` by `(1/n)` and add up all `dw, db` before updating `w, b` and replace `np.random.permutation(n)` by `range(n)` (need to edit the code)

References:
* Backpropagation algorithm derivation in matrix form: https://sudeepraja.github.io/Neural/
