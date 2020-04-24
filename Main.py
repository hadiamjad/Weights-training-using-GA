import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from numpy import exp



def dataPreprocessing():
    # Reading file and seperating dependent and independent variables.
    dataset = pd.read_excel("irisdataset.xlsx")
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    y = y.reshape(150, 1)

    # Econding data set
    labelencoder_y = LabelEncoder()
    y[:, 0] = labelencoder_y.fit_transform(y[:, 0])
    onehotencoder = OneHotEncoder(categorical_features=[0])
    y = onehotencoder.fit_transform(y).toarray()

    # adding the bias in the X dataset
    X = np.insert(X, 0, values=1, axis=1)

    # splitting the dataset 50% for training anf 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    y_train = np.reshape(y_train, (75, 3))
    y_test = np.reshape(y_test, (75, 3))

    return X_train, X_test, y_train, y_test


# ANN
# inputVec 5x75
# inputWeights 10x5
# hiddenWeights 3x11
def ANN(inputVec, inputWeights, hiddenWeights):
    # yhat is 10x75
    inputYhat = np.dot(inputWeights, inputVec)

    # sigmoid
    inputYhat = 1 / (1 + exp(-inputYhat))

    # adding bias to the ouput of input layer
    inputYhat = np.insert(inputYhat.T, 0, values=1, axis=1)
    inputYhat = inputYhat.T

    hiddenYhat = np.dot(hiddenWeights, inputYhat)

    # sigmoid
    hiddenYhat = 1 / (1 + exp(-hiddenYhat))

    # if-else in final output
    hiddenYhat = step_func(hiddenYhat)

    # 3x75
    return hiddenYhat

# for 0/1 final output
def step_func(yhat):
    yhat[yhat <= 0.5] = 0
    yhat[yhat > 0.5] = 1
    return yhat


# Accuracy Calculator
# both are 3x75
def accuracy(y_pred, y_act):
    subt = np.subtract(y_pred.T, y_act.T)
    count = 0
    for row in subt:
        for cell in row:
            if cell != 0:
                count = count + 1
                break
    return count


# main function
def __main__():
   X_train, X_test, y_train, y_test = dataPreprocessing()

   # Random Weights for input and hidden layer
   inputWeights = np.random.rand(10, 5)
   hiddenWeights = np.random.rand(3, 11)
   # calling ANN
   output = ANN(X_train.T, inputWeights, hiddenWeights)


   print(accuracy(output, y_train.T))

   return 0


# calling main function
__main__()
