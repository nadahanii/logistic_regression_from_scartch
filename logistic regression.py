# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 19:26:25 2021

@author: dell
"""
#with multiple trials it was found that the best accuracy while training and the least error
#while testing was found at learning rate = 0.1 and 10000 iterations

import numpy as np
import pandas as pd


data = pd.read_csv("heart.csv")

dataSize = len(data)
trainingSize = int(0.8*dataSize)
testSize = dataSize - trainingSize

trainingData = pd.read_csv("heart.csv",nrows=trainingSize)
testData = pd.read_csv("heart.csv" , skiprows=trainingSize-1 , nrows=testSize)

Y_training=trainingData.iloc[:,[13]]
X_training=trainingData.iloc[:,[3,4,7,9]]

X_test = testData.iloc[:,[3,4,7,9]]
Y_test = testData.iloc[:,[13]]

#normalizing features' scales for training set
for i in range(4):
    min_feature_training= X_training.iloc[:,i].min()
    max_feature_training = X_training.iloc[:,i].max()
    X_training.iloc[:,i]=(X_training.iloc[:,i]-min_feature_training)/(max_feature_training-min_feature_training)


#normalizing features' scales for test set

for i in range(4):
    min_feature_test= X_test.iloc[:,i].min()
    max_feature_test = X_test.iloc[:,i].max()
    X_test.iloc[:,i]=(X_test.iloc[:,i]-min_feature_test)/(max_feature_test-min_feature_test)




class logisticRegression:
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias = None
        self.linear_model = None
        #X is an m-dimensional vector of 
        #size m*n where m is the number of samples and n is the number of features
        #Y is a 1=d vector of size m that has the outputs
    def gradientDescent(self,X,Y):
        #init parameters 
        #to make the multiplication more efficient and between compatible types of data
        X = np.array(X)
        Y = np.array(Y)
       
        #1st dimension of x (rows) is n_samples
        #and 2nd dimension of x (columns) is n_features
        n_samples , n_features = X.shape
        
        #weights are a vector of zeros of size equal to the number of features
        self.weights= np.zeros((n_features,1))
        #to make the shape return (n_features,1) not (n_features,) "put every zero alone"
        #print(self.weights.shape)
        self.bias=0
        hist = []
        #gradientDescent
        for _ in range(self.n_iters): #for each iteration
            self.linear_model = np.dot(X,self.weights) + self.bias
            y_predicted = self.sigmoid(self.linear_model)
            
            #calculate derivatives
            dw = (1/n_samples) * np.dot(X.T,(y_predicted-Y))
            db = (1/n_samples) * np.sum(y_predicted-Y)
            
            #update weights (w and b)
            self.weights -= self.lr * dw
          
            self.bias -= self.lr * db
            hist.append(self.accuracy(Y_true=Y,Y_pred=self.predict(X)))
           
        return hist
          
    def predict(self,X):
       
        linear_model = np.dot(X,self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class
    
    def accuracy(self,Y_true,Y_pred):
        Y_pred = np.array(Y_pred).reshape((len(Y_pred),1))
        accuracy=np.sum(Y_true == Y_pred )/ len(Y_true)
        return accuracy
        
    def sigmoid(self,x):
        return 1/(1+ np.exp(-x))
    
    def error(self,X,Y):
        pred = np.array(self.predict(X))
        real = np.array(Y)
        return np.sum(pred!=real) / len(real)
        
    
regressor = logisticRegression(lr=0.1,n_iters=10000)
x=regressor.gradientDescent(X_training, Y_training) 
print("accuracy while training",x[-1])
print("error in testing",regressor.error(X_test,Y_test))


