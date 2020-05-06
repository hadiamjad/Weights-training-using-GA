import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from numpy import exp

init_cost = 9999999

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
    subt = np.subtract(y_pred.T, y_act)
    count = 0
    for row in subt:
        for cell in row:
            if cell != 0:
                count = count + 1
                break
    return count


# GA for weights training
class Chromosome:
    def __init__(self, w, fitness=0):
        self.fitness = fitness
        self.weig = w

    def __lt__(self, other):
        flag = False
        if self.fitness < other.fitness:
            flag = True
        return flag

# it generates random population of size 'n'
# chromosome is of size 10x5 + 3x11 = 83
def createPopulation(n):
    population = []

    for k in range(n):
        x = np.random.randint(low=1, high=9999999, size=83)
        c = Chromosome(x)
        population.append(c)

    return population

# weights is array of 83 and y is 3x75
# # inputWeights 10x5
# # hiddenWeights 3x11
# # inputVec 5x75
def evaluation(inputVec, weights, y):
    inputWeights, hiddenWeights = np.split(weights, [50])
    inputWeights = np.reshape(inputWeights, (-1, 5))
    hiddenWeights = np.reshape(hiddenWeights, (-1, 11))

    yhat = ANN(inputVec, inputWeights, hiddenWeights)
    cost = accuracy(yhat, y)

    return cost

# Selection
# inputVec 5x75
# weights is array of 83 and y is 3x75
def selection(old_population, inputVec, y):
    global init_cost

    for i in range(len(old_population)):
        old_population[i].fitness = evaluation(inputVec, old_population[i].weig, y)

    old_population.sort(key=lambda individual: individual.fitness)
    fittest = old_population[0].fitness
    if init_cost == 9999999:
        init_cost = fittest

    return fittest, old_population[:20]


# cross over
def crossOver(old_population):
    crossover = []
    for i in range(70):
        t = np.random.randint(0, 20)
        m = old_population[t:t+1]
        t2 = np.random.randint(0, 20)
        m2 = old_population[t2:t2+1]
        p = m[0].weig
        p2 = m2[0].weig
        child = np.zeros(83)
        child[:50] = p[:50]
        child[50:] = p2[50:]
        ind = Chromosome(child)
        crossover.append(ind)
    return crossover[:]

# mutation
def mutation(old_population):
    mutation = []
    for i in range(10):
        t = np.random.randint(0, 20)               # pick a random member of the population
        m = old_population[t:t+1]
        index = np.random.randint(0, 82)            # random selection of a pixel
        p = m[0].weig
        p[index] = np.random.randint(0, 256)
        ind = Chromosome(p)
        mutation.append(ind)
    return mutation

def GeneticAlgorithm(inputVec, y):
    old_population = createPopulation(100)
    new_population = []
    weights = np.zeros(83)
    global init_cost
    fittest = -1
    maxfit = -9999
    i = 0
    while i < 1000:
        fittest, best_old = selection(old_population, inputVec, y)
        new_population.extend(best_old)
        new_population.extend(crossOver(old_population))
        new_population.extend(mutation(old_population))
        if(fittest > maxfit):
            maxfit = fittest
            weights = best_old[0]
        i += 1
        old_population = new_population
        new_population = []
    return fittest, weights

# main function
def __main__():
   X_train, X_test, y_train, y_test = dataPreprocessing()

   fittest, weights = GeneticAlgorithm(X_train.T, y_train)

   print(fittest)

   return 0


# calling main function
__main__()
