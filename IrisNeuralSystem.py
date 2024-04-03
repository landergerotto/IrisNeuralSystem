from random import random, randint
import numpy as np
from math import exp, sqrt, e
from matplotlib import pyplot as plt

class NeuralNet:
    def __init__(self):
        self.layers = []
        # Primeira Camada
        w = [[random() for i in range(4)] for j in range(2)]
        b = [random() for i in range(2)]
        self.layers.append((w, b))
        # Segunda Camada
        w = [[random() for i in range(2)] for j in range(3)]
        b = [random() for i in range(3)]
        self.layers.append((w, b))

    def sigmoid(self, x):
        return 1 / (1 + e ** (-x))

    def predict(self, X):
        # print("X: ",X)
        y = X
        for layer in self.layers:
            w, b = layer
            output = []
            for wj, bj in zip(w, b):
                # print('wj:', wj)
                # print('y:', y)
                # print(type(np.dot(wj, y)))
                neuronExit = self.sigmoid(np.add(np.dot(wj, y), bj))
                output.append(neuronExit)  
            y = output # passando pra próxima layer
        return y
    
    def transformY(self, y):
        if y == 0:
            return [1, 0, 0]
        if y == 1:
            return [0, 1, 0]
        if y == 2:
            return [0, 0, 1]
        
        return [0, 0, 0]

    def score(self, X, Y):
        Y = self.transformY(Y)
        Yn = self.predict(X)
        error = sum([(yi - y) ** 2 for yi, y in zip(Y, Yn)])
        return sqrt(error)
    
    def scoreAll(self, X, Y):
        loss = 0
        for x, y in zip(X, Y):
            loss += self.score(x, y)
        return loss / len(Y)
    
    def fit(self, X, Y, epochs = 100):
        # print('X: ', X)
        scores = []
        for i in range(epochs):
            print("iniciando epoca", i + 1)
            self.epoch(X, Y)
            score = self.scoreAll(X, Y)
            scores.append(score)
            print("score", score)
            print()
        return scores
    
    def epoch(self, X, Y):
        batch = randint(0, len(Y) - 1)
        x = X[batch]
        y = Y[batch]
        y = self.transformY(y)
        pred = self.predict(x)
        helper = []
        w, b = self.layers[1]
        for j in range(len(b)):
            yj = pred[j]
            # print(w[j])
            dB = (yj - y[j]) * yj * (1 - yj)
            b[j] -= 0.25 * dB
            for i in range(len(w[j])):
                helper.append(dB * w[j][i] * x[i] * w[j][i])
                w[j][i] -= 0.25 * dB * w[j][i] * x[i]

        self.layers[1] = (w, b)

        w, b = self.layers[0]
        for j in range(len(b)):
            dB = 0
            yj = self.sigmoid(sum([wj * xj for wj, xj in zip(w[j], x)]) + b[j])
            for lastDevs in helper:
                # dB += helper[j] * yj
                dB += lastDevs * yj
            b[j] -= 0.25 * dB
            for i in range(len(w[j])):
                w[j][i] -= 0.25 * dB * w[j][i] * x[i]
        self.layers[0] = (w, b)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, Y = load_iris(return_X_y = True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

model = NeuralNet()
scores = model.fit(X_train, Y_train)
model.scoreAll(X_test, Y_test)

print(X_test[1])
print(Y_test[1])
print(model.predict(X_test[1]))

print(scores)
plt.plot(scores)
plt.show()