import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# print(os.listdir("../Selected1Project"))
# print(os.listdir("../Selected1Project/flowers"))


daisy_path = "../Selected1Project/flowers/daisy/"
dandelion_path = "../Selected1Project/flowers/dandelion/"
rose_path = "../Selected1Project/flowers/rose/"
sunflower_path = "../Selected1Project/flowers/sunflower/"
tulip_path = "../Selected1Project/flowers/tulip/"

# print(os.listdir(daisy_path))
# print(os.listdir(dandelion_path))
# print(os.listdir(rose_path))
# print(os.listdir(sunflower_path))
# print(os.listdir(tulip_path))

trainLabels = []
data = []
size = 150,150

def readImages(flowerpath,folder):
    
    imagePaths = []
    
    for file in os.listdir(flowerpath):
        if file.endswith("jpg"):
            imagePaths.append(flowerpath+file)
            trainLabels.append(folder)
            img = cv2.imread((flowerpath+file),0)
            im = cv2.resize(img, size)
            data.append(im)
    return imagePaths
    
def showImage(imgPath):
    img = cv2.imread(imgPath)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.axis('off')
    plt.show()

daisyPaths = readImages(daisy_path, 'daisy')
dandelionPaths = readImages(dandelion_path, 'dandelion')
rosePaths = readImages(rose_path, 'rose')
sunflowerPaths = readImages(sunflower_path, 'sunflower')
tulipPaths = readImages(tulip_path, 'tulip')

showImage(daisyPaths[np.random.randint(0,100)])
showImage(dandelionPaths[np.random.randint(0,100)])
showImage(rosePaths[np.random.randint(0,100)])
showImage(sunflowerPaths[np.random.randint(0,100)])
showImage(tulipPaths[np.random.randint(0,100)])

rawData = np.array(data)
print(rawData.shape)

rawData = rawData.astype('float32') / 255.0

X = rawData
z = np.zeros(2159)
o = np.ones(2158)
Y = np.concatenate((z, o), axis = 0).reshape(X.shape[0], 1)

print("\nX shape: " , X.shape)
print("Y shape: " , Y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.15, random_state = 42)
numberOfTrain = xTrain.shape[0]
numberOfTest = xTest.shape[0]

xTrainFlatten = xTrain.reshape(numberOfTrain, xTrain.shape[1] * xTrain.shape[2])
xTestFlatten = xTest.reshape(numberOfTest, xTest.shape[1] * xTest.shape[2])

print("\nX train flatten", xTrainFlatten.shape)
print("X test flatten", xTestFlatten.shape)

x_train = xTrainFlatten.T
x_test = xTestFlatten.T
y_train = yTrain.T
y_test = yTest.T
print("\nx train: ",xTrain.shape)
print("x test: ",xTest.shape)
print("y train: ",yTrain.shape)
print("y test: ",yTest.shape)

def initializeParametersAndLayerSizesNN(x_train, y_train):
    
    parameters = {"weight1": np.random.randn(3, x_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3, 1)),
                  "weight2": np.random.randn(y_train.shape[0], 3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0], 1))}
    
    return parameters

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

def forwardPropagationNN(x_train, parameters):

    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"], A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
              "A1": A1,
              "Z2": Z2,
              "A2": A2}
    
    return A2, cache

def computeCostNN(A2, Y, parameters):
    
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    
    return cost

def backwardPropagationNN(parameters, cache, X, Y):

    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
              "dbias1": db1,
              "dweight2": dW2,
              "dbias2": db2}
    
    return grads

def updateParametersNN(parameters, grads, learning_rate):
    
    parameters = {"weight1": parameters["weight1"] - learning_rate * grads["dweight1"],
                  "bias1": parameters["bias1"] - learning_rate * grads["dbias1"],
                  "weight2": parameters["weight2"] - learning_rate * grads["dweight2"],
                  "bias2": parameters["bias2"] - learning_rate * grads["dbias2"]}
    
    return parameters

def predictNN(parameters, x_test):

    A2, cache = forwardPropagationNN(x_test, parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(A2.shape[1]):
        if A2[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction

def two_layer_neural_network(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    
    cost_list = []
    index_list = []
    
    parameters = initializeParametersAndLayerSizesNN(x_train, y_train)
    for i in range(0, num_iterations):
        A2, cache = forwardPropagationNN(x_train, parameters)
        cost = computeCostNN(A2, y_train, parameters)
        grads = backwardPropagationNN(parameters, cache, x_train, y_train)
        parameters = updateParametersNN(parameters, grads, learning_rate)
        
        if i % 10 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %.4f" %(i, cost))
            
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation = 'vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    y_prediction_test = predictNN(parameters, x_test)
    y_prediction_train = predictNN(parameters, x_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(x_train, y_train, x_test, y_test, learning_rate = 0.01, num_iterations = 100)

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T

def build_classifier():
    classifier = Sequential() 
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()

print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))

input("Press Enter to Exit !")
