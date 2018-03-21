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
    - [ ]Generates a training and a test set of 1, 000 points sampled uniformly in \[0, 1\]², each with a
label 0 if outside the disk of radius 1/√2π and 1 inside,
    - [ ] builds a network with two input units, two output units, three hidden layers of 25 units
    - [ ] trains it with MSE, logging the loss,
    - [ ] computes and prints the final train and the test errors.

## Ideas
- [ ] structure:
    - [ ] framework
        - [ ] network/math_util
            - [ ] tanh,
            - [ ] relu,
            - [ ] other math util functions
        - [ ] fully_connected
            - [ ] ???
            - [ ] forward
            - [ ] backward
        - [ ] loss_util (?)
            - [ ] SGD
            - [ ] MSE
            - [ ] ???
    - [ ] util
        - [ ] data_generation
        - [ ] ???
    - [ ] test
        - [ ] generate data
        - [ ] builds a network
        - [ ] trains it with MSE, logging loss
        - [ ] computes and print final errors (test and train)
    - [ ] ???