# NNNN: a Nth NumPy Neural Network

`NNNN` is a fully connected feedforward neural network with stochastic gradient descent written in Python+NumPy

## Features

* Supports classification and regression
* Depends on `numpy` only
* Weights < 200 LOC

## Usage

```python
from nnnn import NNNN

data_train = ...
data_test = ...

network = NNNN(layers = [64, 16, 10], regression = False)

network.train(data_train, target, iterations = 100, rate = 0.001, alpha = 0.0001)

prediction = network.predict(data_test)
```

(or better, use `sklearn.neural_network.MLPClassifier` or `sklearn.neural_network.MLPRegressor`)

### Initialization

```python
network = NNNN(layers = [64, 16, 10], regression = False)
```

* `layers` is the network structure as a list of int with
  * `layers[0] = n_dimensions` the input dimension
  * `layers[-1] = n_features` the output dimension
* `regression` optimizes the network for regression (`False`) or classification (`True`)

### Training

```python
network.train(data_train, target, iterations = 100, rate = 0.001, alpha = 0.0001)
```

* `data_train` is the input data with `data_train.shape = (n_samples, n_dimensions)`
* `target` is the output target with `target.shape = (n_samples, n_features) or (n_samples,)`
* `iterations` is the number of gradient descent runs
* `rate` is the training rate (default: `0.001`)
* `alpha` is the regularization factor (default: `0.0001`)

### Testing

```python
prediction = network.predict(data_test)
```

* `data_test` is the input data with `data_test.shape = data_train.shape`
* `prediction` is the output prediction with `prediction.shape = target.shape`

## Example

MNIST database with a 3 layers classification network, 1617 training samples and 180 testing samples
(see `examples/nnnn_example.py`)

```
training accuracy = 99%
testing accuracy = 93%
```

Training|Testing
--------|-------
![loss.png](examples/loss.png)|![test.png](examples/test.png)

## Implementation

Activation functions:
* ReLU on the hidden layers
* No activation function on the output layer for regression
* Logistic on the output layer for binary classification
* Softmax on the output layer for multiclass classification

Optimization algorithm:
* Stochastic gradient descent with regularization on the network weights
* Mean squared error loss function for regression
* Mean cross-entropy loss function for classification

## Requirements

`numpy>=1.19.2`

## References

* Backpropagation algorithm derivation in matrix form: https://sudeepraja.github.io/Neural/
* Cross-entropy loss functions and derivations: https://peterroelants.github.io/posts/cross-entropy-logistic/, https://peterroelants.github.io/posts/cross-entropy-softmax/
* Input, weight and bias initialization: https://cs231n.github.io/neural-networks-2/
