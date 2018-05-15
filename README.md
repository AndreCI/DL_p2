# DL_p2
Repo for the second mini project of deep learning.

### Usage

To run the experiment asked in the pdf, run
```
python test.py
```
This will create a network composed of 5 layers, an input layer of size
2 * 25, 3 hidden slayers of size 25 * 25 and an output layer of size 25 * 2
and train it using SGD.

To explore the code a bit more, run
```
python main.py --help
```

Multiple parameters can be used, such as `--hidden_units` or `--load_best_model`.
See `python main --help` for more info.
Note that main uses additional packages but those are used only for utilitaries and no
neural network usage.

### Additional packages

The three following files use additional packages. More precisely, they use
logging, argparse, os, json, sys and matplotlib. However, they can be removed from the project without making it crashes.
Indeed, these packages are used to display, logging and other non deep learning related topics.
```
main.py
data_util.py
configuration.py
```