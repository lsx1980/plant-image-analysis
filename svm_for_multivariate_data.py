"""
Version: 1.5

Summary: This code demonstrates SVM(Support Vector Machine) for classification of multi-dimensional Dataset.
Please refer here a sample dataset of Iris flowers having multiple dimensions i.e.
petal-length, petal-width, sepal-length, sepal-width.

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE

    python3 svm_for_multivariate_data.py -p ~/example/cluster_ml/ -f iris.data



"""


# importing necessary libraries
import argparse
import pandas as pd

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import numpy as np 


def multidimensional_classification(full_path):
    
    # Read dataset into pandas dataframe
    #df = pd.read_csv(full_path, names = ['sepal-len', 'sepal-width', 'petal-len', 'petal-width','target'])
    #data_features = ['sepal-len', 'sepal-width', 'petal-len', 'petal-width']
    
    df = pd.read_excel(full_path, names = ['filename','tassel area', 'tassel area ratio', 'average width', 'average height', 'number of branches', 'average branch length', 'target'])
    
    data_features = ['tassel area', 'tassel area ratio', 'average width', 'average height', 'number of branches', 'average branch length']


    # X -> features, y -> label
    #X = iris.data
    #y = iris.target
    
    # Extract features
    X = df.loc[:, data_features].values

    # Extract target class ID 
    y = df.loc[:, ['target']].values
    
    # Now using scikit-learn model_selection module, split the iris data into train/test data sets
    # keeping 40% reserved for testing purpose and 60% data will be used to train and form model.
    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    # training a linear SVM classifier

    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)

    # model accuracy for X_test  
    accuracy = svm_model_linear.score(X_test, y_test)

    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)
    
    print(accuracy)
    
    '''
    in_data_for_prediction = [[4.9876, 3.348, 1.8488, 0.2], [5.3654, 2.0853, 3.4675, 1.1222], [5.890, 3.33, 5.134, 1.6]]

    p_res = svm_model_linear.predict(in_data_for_prediction)
    
    print('Given first iris is of type:', p_res[0])
    print('Given second iris is of type:', p_res[1])
    print('Given third iris is of type:', p_res[2])
    '''
    
    PCA_analysis(X, y)
    

#colormap mapping
def get_cmap(n, name = 'hsv'):
    """get the color mapping""" 
    #viridis, BrBG, hsv, copper, Spectral
    #Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    #RGB color; the keyword argument name must be a standard mpl colormap name
    return plt.cm.get_cmap(name,n+1)
    
    
    
def PCA_analysis(X, y):
    
    #Dimensionality Reduction using PCA (Principal Component Analysis) Here n_components = 2 means, transform into a 2-Dimensional dataset.
    
    pca = PCA(n_components=2, whiten=True).fit(X)
    
    X_pca = pca.transform(X)
    
    print('explained variance ratio:', pca.explained_variance_ratio_)
    print('Preserved Variance:', sum(pca.explained_variance_ratio_))

    # Print scatter plot to view classification of the simplified dataset
    
    #target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    
    target_names = ['C1', 'C2','C3','C4','C5','C6','C7','C8','C9']
    

    
    colors = get_cmap(len(target_names))

    plt.figure()

    target_list = np.array(y).flatten()
    
    for i, t_name in enumerate(target_names):
        
        color_rgb = colors(i)[:len(colors(i))-1]

        plt.scatter(X_pca[target_list == t_name, 0], X_pca[target_list ==t_name, 1], color = color_rgb, label=t_name)

    plt.legend()
    plt.show()



if __name__ == '__main__':
    
    # loading the iris dataset
    #iris = datasets.load_iris()
    
    # construct the argument and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required = True, help = "path to file")
    ap.add_argument("-f", "--filename", required = True, help = "file name")
    args = vars(ap.parse_args())
    
    
    # parce path to file 
    current_path = args["path"]
    filename = args["filename"]
    
    full_path = current_path + filename

    multidimensional_classification(full_path)
