# DSPN
train.py: training and testing entrance. command: python train/test model_name (random_seed).

model.py: models of DSPN, MLP, Wide & Deep, PNN, DIN, RNN, bi-RNN and DSPNs of ablation studies.

prepare_data.py: data parser, raw data -> training features.

get_data.py: data parser, training features -> numpy arrays for placeholders.

data_iterator.py: generator for training and testing data.

Dice.py: dice and prelu functions.

utils.py: attentions in DIN and other auxiliary functions.

Anonymous Adveriser Dataset is in an information security checking procedure.

Tensorflow 1.9 is used in our experiments.
