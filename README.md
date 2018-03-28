# DL_p2
repo for the second mini project of deep learning

Your framework should import only torch.FloatTensor and torch.LongTensor from pytorch, and
use no pre-existing neural-network python toolbox.


## Question
Should the user be able to stack multiples dense layers without activations layers? yes
Should the user be able to produce a model without a criterion layers? yes
backward>accumulate the gradient wrt the parameters... with rrespect to

## Short TODO

- [ ] Your framework should provide the tools to:
    - [ ] build network combining fully connected layers, tanh and relu
    - [ ] run the forward and backward passes
    - [ ] optimize parameters for SGD and MSE
- [ ] implement a test.py that import your framework and:
    - [x] Generates a training and a test set of 1, 000 points sampled uniformly in \[0, 1\]², each with a
label 0 if outside the disk of radius 1/√2π and 1 inside,
    - [ ] builds a network with two input units, two output units, three hidden layers of 25 units
    - [ ] trains it with MSE, logging the loss,
    - [ ] computes and prints the final train and the test errors.

## Detailed TODO
- [ ] structure:
    - [ ] framework
        - [ ] modules
            - [ ] fully_connected:
                - [ ] init
                - [x] forward
                - [ ] backward
            - [ ] tanh_module
                - [ ] init
                - [x] forward
                - [ ] backward
            - [ ] relu_module
                - [ ] init
                - [x] forward
                - [ ] backward
            - [ ] fully connected handler, i.e. handles multiple layers ???
                - [ ] init
                - [ ] fwd
                - [ ] bwd
            - [ ] loss layers: criterion
                - [x] fwd: loss computation
                - [ ] bwd: gradient computation
        - [ ] network/math_util
            - [ ] Parameter: used for initalization, etc. ?
            - [ ] Linear computation
            - [ ] tanh,
            - [ ] relu,
            - [ ] other activation fct?
            - [ ] other math util functions
    - [ ] util
        - [x] data_generation
        - [ ] ???
    - [ ] test
        - [x] generates data
        - [ ] builds a network
        - [ ] trains it with MSE, logging loss
        - [ ] computes and print final errors (test and train)
    - [ ] ???
- [ ] additional features ?
    - [ ] batch handling
    - [ ] regularization
    - [ ] diverse data set generation
    - [ ] data set and prediction visualization
    - [ ] initializers
    - [ ] other activation functions: leaky relu, sigmoid, etc.