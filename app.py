from torch.optim import Adam
from random import randint, choice
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

class Network(nn.Module):
	def __init__(self, i_, o_, h_):
		super(Network, self).__init__()
		
		# parameters
		self.inputSize = i_
		self.outputSize = h_
		self.hiddenSize = o_

		# weights
		self.W1 = torch.randn(self.inputSize, self.hiddenSize)
		self.W2 = torch.randn(self.hiddenSize, self.outputSize)

	def forward(self, X):
		self.z = torch.matmul(X, self.W1)
		self.z2 = self.sigmoid(self.z)
		self.z3 = torch.matmul(self.z2, self.W2)
		o = self.sigmoid(self.z3)
		return o

	def sigmoid(self, s):
		return 1 / (1 + torch.exp(-s))

	def sigmoidPrime(self, s):
		return s * (1 - s)

	def backward(self, X, y, o):
		self.o_error = y - o
		self.o_delta = self.o_error * self.sigmoidPrime(o)
		self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
		self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
		self.W1 += torch.matmul(torch.t(X), self.z2_delta)
		self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

	def train(self, X, y):
		o = self.forward(X)
		self.backward(X, y, o)

	def saveWeights(self):
		torch.save(self, "NN")

	def predict(self):
		print ("Predicted data based on trained weights: ")
		print ("Input (scaled): \n" + str(xPredicted))
		print ("Output: \n" + str(self.forward(xPredicted)))

class Snake:
	def __init__(self, tilew, tileh):
		self.tilew = tilew
		self.tileh = tileh
		self.parts = []
		self.reset()

	def __getitem__(self, key):
		return self.parts[key]
	
	def __setitem__(self, key, value):
		self.parts[key] = value

	def reset(self):
		initx = randint(self.tilew // 4, 3 * self.tilew // 4)
		inity = randint(self.tileh // 4, 3 * self.tilew // 4)
		direction = choice([(0,1),(0,-1),(1,0),(-1,0)])
		f = lambda x, y : (x[0] + y[0], x[1] + y[1])
		self.parts = [(initx, inity), \
						f((initx, inity),direction), \
						f(f((initx, inity),direction), direction)]

	def update(self, new_x, new_y):
		
		self.parts.append((new_x, new_y))
		del self.parts[0]

class App(pyglet.window.Window):
	def __init__(self, tile, apple, death = "data/death.jpeg", width = 510, height = 510, offset = 5, tilesize = 10):
		super().__init__(width, height)

		self.width = width
		self.height = height
		self.offset = offset
		self.tilesize = tilesize
		self.tilew = (self.width - self.offset * 2) // self.tilesize
		self.tileh = (self.height - self.offset * 2) // self.tilesize

		self.snake = Snake(self.tilew, self.tileh)
		self.food = (randint(0, self.tilew - 1), \
					randint(0, self.tileh - 1))

		# self.tensor = torch.zeros([self.tilew, self.tileh], dtype = torch.int32)
		self.nn = Network(3, 10, 1)

		self.tile = pyglet.resource.image(tile)
		self.tile.width = self.tilesize - 2
		self.tile.height = self.tilesize - 2

		self.apple = pyglet.resource.image(apple)
		self.apple.width = self.tilesize - 2
		self.apple.height = self.tilesize - 2

		self.death = pyglet.resource.image(death)
		self.death.width = self.width
		self.death.height = self.height

		self.keys = {"UP" : 0, "DOWN" : 0, "LEFT" : 0, "RIGHT" : 0}

		pyglet.gl.glClearColor(0.1, 0.1, 0.1, 1)
		pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
		pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
		pyglet.clock.schedule_interval(self.update, 1 / 20)


	def on_key_press(self, symbol, modifiers):
		if symbol == key.ESCAPE:
			exit()
		if symbol == key.UP:
			self.keys["UP"] = 1
		if symbol == key.DOWN:
			self.keys["DOWN"] = 1
		if symbol == key.LEFT:
			self.keys["LEFT"] = 1
		if symbol == key.RIGHT:
			self.keys["RIGHT"] = 1

	def on_key_release(self, symbol, modifiers):
		if symbol == key.UP:
			self.keys["UP"] = 0
		if symbol == key.DOWN:
			self.keys["DOWN"] = 0
		if symbol == key.LEFT:
			self.keys["LEFT"] = 0
		if symbol == key.RIGHT:
			self.keys["RIGHT"] = 0

	def on_draw(self):
		self.clear()
		self.draw()

	def draw(self):
		# drawing grid
		for i in range(self.width // self.tilesize):
			pyglet.graphics.draw_indexed(2, pyglet.gl.GL_LINES,
				[0, 1],
				('v2i', (self.offset + i * self.tilesize, self.offset,
						self.offset + i * self.tilesize, self.height - self.offset)),
				('c4B', (255, 255, 255, 80, \
						255, 255, 255, 80))
			)

		for i in range(self.height // self.tilesize):
			pyglet.graphics.draw_indexed(2, pyglet.gl.GL_LINES,
				[0, 1],
				('v2i', (self.offset, self.offset + i * self.tilesize,
						self.width - self.offset, self.offset + i * self.tilesize)),
				('c4B', (255, 255, 255, 80, \
						255, 255, 255, 80))
				)

		#drawing snake
		for part in self.snake.parts:
			self.tile.blit(self.offset + part[0] * self.tilesize + 1, \
							self.offset + part[1] * self.tilesize + 1)

		#drawing food
		self.apple.blit(self.offset + self.food[0] * self.tilesize + 1, \
						self.offset + self.food[1] * self.tilesize + 1)

	def update(self, dt):
		### Here, AI will have to choose which direction to go,
		### via changing the keys dictionary.
		# Shitty....
		res = float(self.nn(torch.tensor((self.snake.parts[-1][0], self.snake.parts[-1][1], self.snake.parts[-2][0]), dtype=torch.float))[0])
		if (res <= 0.25):
			self.keys = {"UP" : 1, "DOWN" : 0, "LEFT" : 0, "RIGHT" : 0}
		elif (res <= 0.5):
			self.keys = {"UP" : 0, "DOWN" : 1, "LEFT" : 0, "RIGHT" : 0}
		elif (res <= 0.75):
			self.keys = {"UP" : 0, "DOWN" : 0, "LEFT" : 1, "RIGHT" : 0}
		elif (res <= 1):
			self.keys = {"UP" : 0, "DOWN" : 0, "LEFT" : 0, "RIGHT" : 1}

		if (any(self.keys.values())):
			new_x = self.snake[-1][0] + self.keys["RIGHT"] - self.keys["LEFT"]
			new_y = self.snake[-1][1] + self.keys["UP"] - self.keys["DOWN"]
			if (new_x < 0 or new_x >= self.tilew or \
				new_y < 0 or new_y >= self.tileh or \
				(new_x, new_y) in self.snake.parts):
				self.death.blit(0,0)
				self.snake.reset()
			else:
				self.snake.update(new_x, new_y)
		if (self.snake.parts[-1] == self.food):
			self.snake.parts.append(self.food)

			self.food = (randint(0, self.tilew - 1), \
						randint(0, self.tileh - 1))
