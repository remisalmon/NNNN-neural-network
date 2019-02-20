# NNNN: a Nth NumPy Neural Network

nnnn.py is a feed-forward neural network with stochastic gradient descent written in Python

## Comments

Notations:  
* `x, y` are vectors of dimensions `(d, 1), (c, 1)`
* `X, Y` are matrices of dimensions `(d, n), (c, n)`
* `d` is the input data dimensions, `c` is the number of classes, `n` is the number of training samples
* `y, Y` are training data and `y_hat, Y_hat` are network output data

Remarks:
* In `df_cost()`: for the stochastic gradient descent, the sum in the cost function `f` is removed since only `y_hat, y` are evaluated instead of the whole training sample `Y_hat, Y` (default algorithm, no need to edit the code)
* To use the batch gradient descent, multiply `df` by `(1/n)` and add up all `dw, db` before updating `w, b` (need to edit the code)

References:
* Backpropagation algorithm derivation in matrix form: https://sudeepraja.github.io/Neural/
