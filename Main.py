import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder



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

    # splitting the dataset 50% for training anf 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    y_train = np.reshape(y_train, (75, 3))
    y_test = np.reshape(y_test, (75, 3))

    return X_train, X_test, y_train, y_test


# main function
def __main__():
   X_train, X_test, y_train, y_test = dataPreprocessing()


   return 0


# calling main function
__main__()
