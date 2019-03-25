# CNN_SVHN

<h1>Training a Convolutional Neural Network (CNN) on the SVHN Dataset (Format 2) to classify house numbers</h1>

Description: Using Keras Sequential interface, training a CNN on the SVHN Dataset (Format 2) to classify house numbers.

Architecture:

Img

<h2>Solution Steps:</h2>

<h1>List of packages</h1>
    import os
    import itertools
    import numpy as np
    import scipy.io
    import pandas as pd
    import seaborn as sns

    import keras
    from keras.models import Sequential
    from keras.utils import np_utils
    from keras.layers import Dense, Dropout, Activation, BatchNormalization,Flatten,Conv2D
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.optimizers import SGD, Adadelta, Adagrad

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt

    %matplotlib inline
    plt.rcParams['figure.figsize'] = (12.0, 8.0)

<h1></h1>
<h1></h1>
<h1></h1>
<h1></h1>
<h1></h1>
<h1></h1>
<h1></h1>

<h2>Dataset Source</h2>
http://ufldl.stanford.edu/housenumbers/


<h2>Files</h2>
<li>cnn_svhn.py includes codes to load SVHN dataset, train models, draw accuracy and loss graphs and predict label of test dataset.</li>
