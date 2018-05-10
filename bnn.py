import theano
import pymc3 as pm
import theano.tensor as T
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from warnings import filterwarnings
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons
data = pd.read_csv("vectors.csv")
data_train = data[: int(data.shape[0] * 0.9)]
data_test = data[int(0.9 * data.shape[0]) : data.shape[0]]
y_test = data_test.ix[:,11]
x_test = data_test.ix[:,range(11)]
y_train = data_train.ix[:,11]
x_train = data_train.ix[:,range(11)]
y = data.ix[:,11]
x = data.ix[:,range(11)]
ann_input = theano.shared(x_train.as_matrix())
ann_output = theano.shared(y_train.as_matrix())
n_hidden = 10

# Initialize random weights between each layer
init_1 = np.random.randn(x.shape[1], n_hidden)
init_2 = np.random.randn(n_hidden, n_hidden)
init_3 = np.random.randn(n_hidden, n_hidden)
init_4 = np.random.randn(n_hidden, n_hidden)
init_out = np.random.randn(n_hidden)

with pm.Model() as neural_network:
    # Weights from input to hidden layer
    weights_in_1 = pm.Normal('w_in_1', 0, sd=1,
                             shape=(x.shape[1], n_hidden),
                             testval=init_1)

    # Weights from 1st to 2nd layer
    weights_1_2 = pm.Normal('w_1_2', 0, sd=1,
                            shape=(n_hidden, n_hidden),
                            testval=init_2)
    weights_2_3 = pm.Normal('w_2_3', 0, sd=1,
                            shape=(n_hidden, n_hidden),
                            testval=init_3)
    weights_3_4 = pm.Normal('w_3_4', 0, sd=1,
                            shape=(n_hidden, n_hidden),
                            testval=init_4)
    # Weights from hidden layer to output
    weights_4_out = pm.Normal('w_4_out', 0, sd=1,
                              shape=(n_hidden,),
                              testval=init_out)
    # Build NN
    act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
    act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
    act_3 = pm.math.tanh(pm.math.dot(act_2, weights_2_3))
    act_4 = pm.math.tanh(pm.math.dot(act_3, weights_3_4))
    act_out = pm.math.sigmoid(pm.math.dot(act_4, weights_4_out))
    #Since it is a clasification problem, use Bernoulli
    out = pm.Bernoulli('out',
                       act_out,
                       observed=ann_output)

with neural_network:
    inference = pm.ADVI()
    approx = pm.fit(n=30000, method=inference)
trace = approx.sample(draws=50000)
ann_input.set_value(x_test)
ann_output.set_value(y_test)

# Creater posterior predictive samples
ppc = pm.sample_ppc(trace, model=neural_network, samples=1000)

pred = ppc['out'].mean(axis=0) > 0.5

print(' {}%'.format((y_test == pred).mean() * 100))