# DL_p2
repo for the second mini project of deep learning

Your framework should import only torch.FloatTensor and torch.LongTensor from pytorch, and
use no pre-existing neural-network python toolbox.


## Question
Should the user be able to stack multiples dense layers without activations layers? yes
Should the user be able to produce a model without a criterion layers? yes
backward>accumulate the gradient wrt the parameters... with rrespect to


Import only torch.FloatTensor: does that mean that we can't use torch.mm? or torch.from_numpy or torch.tanh?
Ofc

How well is the model suppose to perform? Are we supposed to add regularization and such?
~10%. Not regu is asked, but we can add it.

It is not specified wheter or not we can use softmax classifier, which should change the performance of the model tremendously
You can, but not needed.

Is the type of activation layer a kind of parameter to optimize?
Not really. Display info for all possibilities or best one, whatever.

In what type of form must the rapport be? A documentation from the code, or rather an theoretical explanation of how it works?
Journal like, explain what I've done and why. Present examples on how to use the code

## Short TODO

- [ ] Your framework should provide the tools to:
    - [x] build network combining fully connected layers, tanh and relu
    - [x] run the forward and backward passes
    - [ ] optimize parameters for SGD and MSE
    - [ ] add comments to everything
- [x] implement a test.py that import your framework and:
    - [x] Generates a training and a test set of 1, 000 points sampled uniformly in \[0, 1\]², each with a
label 0 if outside the disk of radius 1/√2π and 1 inside,
    - [x] builds a network with two input units, two output units, three hidden layers of 25 units
    - [x] trains it with MSE, logging the loss,
    - [x] computes and prints the final train and the test errors.

## Detailed TODO
- [ ] structure:
    - [ ] framework
        - [ ] modules
            - [x] fully_connected:
                - [x] init
                - [x] forward
                - [X] backward
                - [x] compute gradient
                - [x] apply gradient
            - [ ] tanh_module
                - [x] forward
                - [x] backward
            - [ ] relu_module
                - [x] forward
                - [x] backward
            - [x] fully connected handler, i.e. handles multiple layers ???
                - [x] dense layer handler
                - [x] activation handler
            - [x] loss layers: criterion
                - [x] fwd: loss computation
                - [x] bwd: gradient computation
        - [ ] network/math_util
            - [ ] Parameter: used for initalization, etc. ?
            - [x] Linear computation
            - [x] Xavier init
            - [ ] other activation fct?
            - [ ] other math util functions
    - [x] util
        - [x] data_generation
        - [ ] ???
    - [x] test
        - [x] generates data
        - [x] builds a network
        - [x] trains it with MSE, logging loss
        - [x] computes and print final errors (test and train)
    - [ ] ???
- [ ] additional features ?
    - [ ] batch handling
    - [ ] regularization
    - [ ] diverse data set generation
    - [x] data set and prediction visualization
    - [ ] initializers
    - [ ] other activation functions: leaky relu, sigmoid, etc.

### Additional infos
following the instructions and training setup from the exercises you should be able to get less than 10% error on the test set with an MLP with tanh activations (for the 2nd miniproject). (slack)
