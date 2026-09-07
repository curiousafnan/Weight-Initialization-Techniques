import random
import math
import numpy as np
from src.activation import *
from src.initialization import *


class Loss:
    def loss(self,y,y_pred): raise NotImplementedError
    def derivative(self,y,y_pred): raise NotImplementedError
    def __call__(self,y,y_pred): return self.derivative(y,y_pred)

class SE(Loss):
    def loss(self, y,y_pred):
        return np.square(y-y_pred)
    def derivative(self, y,y_pred):
        return -2 * (y-y_pred)

class MSE(Loss):
    def loss(self, y,y_pred):
        return np.mean(np.square(y-y_pred))
    def derivative(self, y,y_pred):
        return 2 * (y_pred - y) / y.size

class Layer:
    def __init__(self,no_perceptrons:int,activation:Activation,initializer: Initializer,input:int=5,lr=0.01):
        self.activation = activation
        self.initializer= initializer
        self.no_perceptrons = no_perceptrons
        self.input = input
        self.W = []
        self.b = []
        self.next_layer = None
        self.lr = lr

    def compile(self):
        self.W = self.initializer(self.input,self.no_perceptrons)
        self.b = np.zeros(self.no_perceptrons,)

    def forward(self,X):
        X = np.asarray(X, dtype=float)
        self._last_X = X
        self._last_Z = X @ self.W.T + self.b
        self._last_activation = self.activation.forward(self._last_Z)
        return self._last_activation

    def backward(self, dL_dA,weights_update:bool):
        dL_dZ = dL_dA * self.activation.derivative(self._last_Z)
        dL_dW = np.outer(dL_dZ , self._last_X)
        dL_dB = dL_dZ
        dL_dX = dL_dZ @ self.W
        self.dL_dW = dL_dW
        self.dL_dB = dL_dB
        if weights_update:
            self.W = self.W - self.lr * dL_dW
            self.b = self.b - self.lr * dL_dB
        return dL_dX



class ANN:
    def __init__(self,input,epochs,loss: Loss,*args:Layer):
        self.root = None
        self.input = input
        self.epochs = epochs
        self.loss = loss
        self.layers = [layer for layer in args]

    def compile(self):
        if self.input<=0:
            raise ValueError("The input should greater than zero.")
        for idx, layer in enumerate(self.layers):
            if idx ==0:
                self.root = layer
                layer.input = self.input
            else:
                last_layer = self.layers[idx-1]
                prev_perceptrons = last_layer.no_perceptrons
                last_layer.next_layer = layer
                layer.input = prev_perceptrons
            layer.compile()

    def forward(self,X:np):
        y = X
        for layer in self.layers:
            y=layer.forward(y)
        return y

    def backprop(self,dL_dA,weights_update:bool=True):
        for layer in reversed(self.layers):
            dL_dA = layer.backward(dL_dA,weights_update)
        return dL_dA

    def fit(self, X, y):
        loss_history = []
        for epoch in range(self.epochs):
            for x_i,y_i in zip(X,y):
                y_pred = self.forward(x_i)
                dL_dA = self.loss(y_i,y_pred)
                loss = self.loss.loss(y_i,y_pred)
                self.backprop(dL_dA)
            loss_history.append(loss)
        return loss_history



if __name__ == "__main__":
    layer1 = Layer(no_perceptrons=4, activation=Sigmoid(),lr=0.5)
    layer2 = Layer(no_perceptrons=1, activation=Sigmoid(),lr=0.5)
    ann = ANN(2,5000,MSE(), layer1, layer2)
    ann.compile()
    # output = ann.forward(np.array([0, 1]))
    # print(output)
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([[0],[1],[1],[0]], dtype=float)
    ann.fit(X, y)
    for x in [[0,0], [0,1], [1,0], [1,1]]:
        output = ann.forward(np.array(x))
        print(f"Input: {x} → Output: {output}")

    grid = np.random.rand(10000, 2)
    print(ann.forward(grid).shape)
    
