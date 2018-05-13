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

To explore the code a bit mode, run
```
python main.py --help
```

Multiple parameters can be used, such as `--hidden_units` or `--load_best_model`.
See `python main --help` for more info. Not that main uses additional packages
such as logging, json or matplotlib but those are used only for utilitaries and no
neural network usage

