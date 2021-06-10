from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pyglet.window import key
from app import *

import pandas as pd
import pyglet
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# X = torch.tensor(([0, 0, 0],[1, 0, 0],[0, 0, 0]), dtype=torch.float)
# y = torch.tensor(([0.1], [0.26], [0.51]), dtype=torch.float)
# xPredicted = torch.tensor(([0, 1, 0]), dtype=torch.float)

# # X_max, _ = torch.max(X, 0)
# # xPredicted_max, _ = torch.max(xPredicted, 0)

# # X = torch.div(X, X_max)
# # xPredicted = torch.div(xPredicted, xPredicted_max)
# print(X, xPredicted, y)
# # y = y / 100

# class Network(nn.Module):
# 	def __init__(self, i_, h_, o_):
# 		super(Network, self).__init__()
		
# 		# parameters
# 		self.inputSize = i_
# 		self.outputSize = o_
# 		self.hiddenSize = h_

# 		# weights
# 		self.W1 = torch.randn(self.inputSize, self.hiddenSize)
# 		self.W2 = torch.randn(self.hiddenSize, self.outputSize)

# 	def forward(self, X):
# 		self.z = torch.matmul(X, self.W1)
# 		self.z2 = self.sigmoid(self.z)
# 		self.z3 = torch.matmul(self.z2, self.W2)
# 		o = self.sigmoid(self.z3)
# 		return o

# 	def sigmoid(self, s):
# 		return 1 / (1 + torch.exp(-s))

# 	def sigmoidPrime(self, s):
# 		return s * (1 - s)

# 	def backward(self, X, y, o):
# 		self.o_error = y - o
# 		self.o_delta = self.o_error * self.sigmoidPrime(o)
# 		self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
# 		self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
# 		self.W1 += torch.matmul(torch.t(X), self.z2_delta)
# 		self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

# 	def train(self, X, y):
# 		o = self.forward(X)
# 		self.backward(X, y, o)

# 	def saveWeights(self):
# 		torch.save(self, "NN")

# 	def predict(self):
# 		print ("Predicted data based on trained weights: ")
# 		print ("Input (scaled): \n" + str(xPredicted))
# 		print ("Output: \n" + str(self.forward(xPredicted)))


# NN = Network(3,10,1)
# for i in range(1000):
# 	print("#" + str(i) + " " + str(torch.mean((y - NN(X))**2).detach().item()))
# 	NN.train(X, y)

# NN.saveWeights()
# NN.predict()

window = App("data/snake.jpeg", "data/food.png")
pyglet.app.run()
