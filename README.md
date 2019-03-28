# CNN_SVHN

<h1>Training a Convolutional Neural Network (CNN) on the SVHN Dataset (Format 2) to classify house numbers</h1>

Description: Using Keras packages and Sequential interface to train a CNN model on the SVHN Dataset (Format 2) to classify house numbers.

Architecture:

    CNN model to classify house numbers consists of following layers:

    1. First, convolutional input layer:
    Layer with 32 units, input shape (32, 32, 3) and RELU as an activation function. Batch 
    normalization to normalize first layer output data.

    2. Second, convolutional hidden layer:
    Layer with 32 units and RELU as an activation function. Applied batch normalization, 2D max 
    polling of (2,2) and dropout of 0.25 as additional layers.

    3. Third, convolutional hidden layer:
    Layer with 64 units and RELU as an activation function and applied batch normalization.

    4. Fourth, convolutional hidden layer:
    Layer with 64 units and RELU as an activation function. Applied batch normalization, 2D max 
    polling of (2,2) and dropout of 0.25 as additional layers.

    5. Flatten Layer:
    Added flatten layer with batch normalization to transform data before feeding to dense layer

    6. First Dense Layer:
    Added dense layer with 512 units and RELU as an activation function. Applied batch  
    normalization and dropout of 0.5 as additional layers.

    7. Second Dense Layer (Output Layer):
    Added output layer as a dense layer with 10 units and softmax as an activation function to 
    predict house number.



![](Architecture.JPG)

<h2>Solution Steps:</h2>

<h3>List of packages</h3>

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

<h3>Load Street View House Numbers (SVHN) Dataset (Format 2)</h3>
<p><p>Download all three SVHN Dataset (Format 2) files from (http://ufldl.stanford.edu/housenumbers/) source</p></p>

    train_dataset = scipy.io.loadmat('train_32x32.mat') 
    train_extra_dataset = scipy.io.loadmat('extra_32x32.mat')
    test_dataset = scipy.io.loadmat('test_32x32.mat')

<h3>Preprocess Dataset</h3>

    # Separate actual image data and label
    X_train = train_dataset['X']
    y_train = train_dataset['y']
    X_extra_train = train_extra_dataset['X']
    y_extra_train = train_extra_dataset['y']
    X_test = Test['X']
    y_test = Test['y']

    # Print shape of the dataset
    print('Training set: ', X_train.shape, y_train.shape)
    print('Training extra set: ', X_extra_train.shape, X_extra_train.shape)
    print('Testing set: ', X_test.shape, y_test.shape)

    # Encode target column
    X_train = X_train.astype('float32')
    X_extra_train = X_extra_train.astype('float32')
    X_test = X_test.astype('float32')

    # Scale data instance values between 0 to 1, before feeding to the neural network model
    X_train /= 255
    X_extra_train /= 255
    X_test /= 255

    X_train = X_train[np.newaxis,...]
    X_train = np.swapaxes(X_train,0,4).squeeze()

    X_extra_train = X_extra_train[np.newaxis,...]
    X_extra_train = np.swapaxes(X_extra_train,0,4).squeeze()

    X_test = X_test[np.newaxis,...]
    X_test = np.swapaxes(X_test,0,4).squeeze()


    np.place(y_train,y_train == 10,0)
    np.place(y_extra_train,y_extra_train == 10,0)
    np.place(y_test,y_test == 10,0)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_extra_train = keras.utils.to_categorical(y_extra_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

<h3>Model settings</h3>

    batch_size = 128
    nb_classes = 10
    nb_epoch = 20

<h3>Create sequential model object</h3>
    
    # create Sequential model object
    model = Sequential()

<h3>Model architecture setting</h3>
 

<h3>Model training </h3>

<h3>Performance matrix on Training and Validation dataset</h3>

<h3>Graph: Training and Validation dataset performance</h3>

<h3>Evaluate model performance on Test dataset</h3>

<h3>Make predictions on Test Dataset</h3>

<h3>Graph: Plot the first X test images, their predicted label, and the true label</h3>

<h3>Save entire model to a HDF5 file</h3>

<h2>Dataset Source</h2>
http://ufldl.stanford.edu/housenumbers/


<h2>Files</h2>
<li>cnn_svhn.py includes codes to load SVHN dataset, train models, draw accuracy and loss graphs and predict label of test dataset.</li>
