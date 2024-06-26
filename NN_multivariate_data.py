"""
Version: 1.5

Summary: Build a Neural Network (Multi-class Classification) for Tassel shape calssification

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 NN_multivariate_data.py -p ~/example/cluster_ml/ -f trait_part.xlsx

"""


# importing necessary libraries
# for reading data
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import keras

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



def multidimensional_classification(full_path):
    
    """Neural Network (Multi-class Classification) for Tassel shape calssification
    
    Inputs: 
    
        full_path: full path of the training excel file

    Returns:
    
        print out accuracy and confusion matrix
        
    """
    
    # Read dataset into pandas dataframe
 
    #df = pd.read_excel(full_path, names = ['filename','tassel area', 'tassel area ratio', 'average width', 'average height', 'number of branches', 'average branch length', 'tassel_type'])
    
    df = pd.read_excel(full_path, names = ['tassel area', 'tassel area ratio', 'average width', 'average height', 'number of branches', 'average branch length', 'species'])
    data_features = ['tassel area', 'tassel area ratio', 'average width', 'average height', 'number of branches', 'average branch length']

    # shuffle the dataset! 
    df = df.sample(frac=1).reset_index(drop=True)

     # Extract features
    #X = df.loc[:, data_features].values

    # Extract target class ID 
    #Y = df.loc[:, ['species']].values
    

    # split into X and Y
    Y = df['species']
    X = df.drop(['species'], axis=1)
    
    #Y = df['tassel_type']
    #X = df.drop(['tassel_type'], axis = 1)

    #print(X.shape)
    #print(Y.shape)

    # convert to numpy arrays
    X = np.array(X)

    # show Y
    Y.head()

    # work with labels
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    #print(encoded_Y)

    #print(dummy_y)

    # build a model
    model = Sequential()
    model.add(Dense(16, input_shape = (X.shape[1],), activation = 'relu')) # input shape is (features,)
    model.add(Dense(6, activation = 'softmax'))
    model.summary()

    # compile the model, this is different instead of binary_crossentropy (for regular classification)
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
              

    # early stopping callback
    # This callback will stop the training when there is no improvement in  
    # the validation loss for 10 consecutive epochs.  
    es = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, restore_best_weights = True) 

    # update the model to fit call
    history = model.fit(X, dummy_y, callbacks = [es], epochs = 8000000, batch_size = 10, shuffle = True, validation_split = 0.2, verbose = 1)

    history_dict = history.history

    # learning curve accuracy
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    # loss
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    # range of X (no. of epochs)
    epochs = range(1, len(acc) + 1)

    # plot
    plt.figure()
    # "r" is for "solid red line"
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    # b is for "solid blue line"
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

     # save plot result
    result_file = (current_path + 'Accuracy_plot.png')
    plt.savefig(result_file)

    preds = model.predict(X) # see how the model did!
    print(preds[0]) # i'm spreading that prediction across three nodes and they sum to 1
    print(np.sum(preds[0])) # sum it up! Should be 1

    # Almost a perfect prediction
    # actual is left, predicted is top
    # names can be found by inspecting Y
    matrix = confusion_matrix(dummy_y.argmax(axis = 1), preds.argmax(axis = 1))


    # more detail on how well things were predicted
    print(classification_report(dummy_y.argmax( axis = 1), preds.argmax(axis = 1)))



if __name__ == '__main__':
    
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to file")
    ap.add_argument("-f", "--filename", required = True, help = "file name")
    args = vars(ap.parse_args())
    
    
    # parce path to file 
    current_path = args["path"]
    filename = args["filename"]
    
    # full path to data file
    full_path = current_path + filename
    
     # classification 
    multidimensional_classification(full_path)
