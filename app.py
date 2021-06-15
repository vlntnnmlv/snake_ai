from torch.optim import Adam
from random import randint, choice
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from collections import namedtuple
from pyglet.window import key
from app import *

import pandas as pd
import pyglet
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
							'state', 'action', 'next_state', 'reward')

class DQNAgent:
	def __init__(self, epsilon = 1):
		self.epsilon = epsilon

	def get_state(self, game):
		s = 1
		if game.snake.direction == (0, 1):
			s *= 1
		elif game.snake.direction == (0, -1):
			s *= 2
		elif game.snake.direction == (1, 0):
			s *= 3
		else:
			s *= 4

		if game.food.position[0] > game.snake.parts[-1][0]:
			s *= 2
		if game.food.position[1] > game.snake.parts[-1][1]:
			s *= 2

		if game.snake.parts[-1][0] == 0 or game.snake.parts[-1][0] == game.tilew - 1:
			s *= 2
		if game.snake.parts[-1][1] == 0 or game.snake.parts[-1][1] == game.tileh - 1:
			s *= 2

		return s

class Snake:
	def __init__(self, tilew, tileh):
		self.tilew = tilew
		self.tileh = tileh
		self.parts = []
		self.reset()
		self.direction = choice([(0,1),(0,-1),(1,0),(-1,0)])

	def __getitem__(self, key):
		return self.parts[key]
	
	def __setitem__(self, key, value):
		self.parts[key] = value

	def reset(self):
		initx = randint(self.tilew // 4, 3 * self.tilew // 4)
		inity = randint(self.tileh // 4, 3 * self.tilew // 4)
		direction = choice([(0,1),(0,-1),(1,0),(-1,0)])
		f = lambda x, y : (x[0] + y[0], x[1] + y[1])
		self.direction = choice([(0,1),(0,-1),(1,0),(-1,0)])
		self.parts = [(initx, inity), \
						f((initx, inity),direction), \
						f(f((initx, inity),direction), direction)]

	def move(self):
		self.parts.append((self[-1][0] + self.direction[0], \
							self[-1][1] + self.direction[1]))
		del self.parts[0]
		if (self.parts[-1][0] < 0 or self.parts[-1][0] >= self.tilew or \
				self.parts[-1][1] < 0 or self.parts[-1][1] >= self.tileh):
			self.reset()

	def turn(self, new_dir):
		if (new_dir[0] * self.direction[0] < 0 or \
			new_dir[1] * self.direction[1] < 0):
			self.reset()
		self.direction = new_dir

	def grow(self, food):
		self.parts.append(food.position)

class Food:
	def __init__(self, tilew, tileh):
		self.tilew = tilew
		self.tileh = tileh
		self.reset()

	def reset(self):
		self.position = (randint(0, self.tilew - 1), \
						randint(0, self.tileh - 1))
		return self.position

class Game(pyglet.window.Window):
	def __init__(self, tile, apple, width = 510, height = 510, offset = 5, tilesize = 10):
		super().__init__(width, height)

		self.width = width
		self.height = height
		self.offset = offset
		self.tilesize = tilesize
		self.tilew = (self.width - self.offset * 2) // self.tilesize
		self.tileh = (self.height - self.offset * 2) // self.tilesize

		self.snake = Snake(self.tilew, self.tileh)
		self.food = Food(self.tilew, self.tileh)

		self.tile = pyglet.resource.image(tile)
		self.tile.width = self.tilesize - 2
		self.tile.height = self.tilesize - 2

		self.apple = pyglet.resource.image(apple)
		self.apple.width = self.tilesize - 2
		self.apple.height = self.tilesize - 2

		pyglet.gl.glClearColor(0.1, 0.1, 0.1, 1)
		pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
		pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
		pyglet.clock.schedule_interval(self.update, 1 / 15)

	def on_key_press(self, symbol, modifiers):
		if symbol == key.ESCAPE:
			exit()
		if symbol == key.UP:
			self.snake.turn((0, 1))
		if symbol == key.DOWN:
			self.snake.turn((0, -1))
		if symbol == key.LEFT:
			self.snake.turn((-1, 0))
		if symbol == key.RIGHT:
			self.snake.turn((1, 0))
 
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
		self.apple.blit(self.offset + self.food.position[0] * self.tilesize + 1, \
						self.offset + self.food.position[1] * self.tilesize + 1)

	def update(self, dt):
		### Here, AI will have to choose which direction to go,
		### via changing the keys dictionary.

		self.snake.move()
		if (self.snake.parts[-1] == self.food.position):
			self.snake.grow(self.food)
			self.food.reset()
		
