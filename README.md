# DL_p2
repo for the second mini project of deep learning

Your framework should import only torch.FloatTensor and torch.LongTensor from pytorch, and
use no pre-existing neural-network python toolbox.

## TODO

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

## Ideas
- [ ] structure:
    - [ ] framework
        - [ ] modules
            - [ ] fully_connected:
                - [ ] ???
                - [ ] forward
                - [ ] backward
            - [ ] tanh_module
            - [ ] relu_module
            - [ ] fully connected handler, i.e. handles multiple layers ???
                - [ ] ???
                - [ ] fwd
                - [ ] bwd
            - [ ] loss layers: criterion
                - [ ] fwd: loss computation
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
        - [ ] generates data
        - [ ] builds a network
        - [ ] trains it with MSE, logging loss
        - [ ] computes and print final errors (test and train)
    - [ ] ???
- [ ] additional features
    - [ ] batch handling
    - [ ] regularization
    - [ ] diverse data set generation
    - [ ] data set and prediction visualization
    - [ ] initializers