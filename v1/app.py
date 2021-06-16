from torch.optim import Adam
from random import randint, choice
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pyglet.window import key

import pandas as pd
import collections
import pyglet
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DQN import *
from game import *

class App(pyglet.window.Window):
	def __init__(self, tile, apple, agent, width = 410, height = 410, offset = 5, tilesize = 20):
		super().__init__(width, height)
		self.game = Game(width, height, offset, tilesize)

		self.game.agent = DQNAgent()
		self.game.agent.load_state_dict(torch.load(agent))
		self.game.agent.eval()

		self.tile = pyglet.resource.image(tile)
		self.tile.width = self.game.tilesize - 2
		self.tile.height = self.game.tilesize - 2

		self.apple = pyglet.resource.image(apple)
		self.apple.width = self.game.tilesize - 2
		self.apple.height = self.game.tilesize - 2

		pyglet.gl.glClearColor(0.1, 0.1, 0.1, 1)
		pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
		pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
		pyglet.clock.schedule_interval(self.update, 1 / 20)
 
	def on_draw(self):
		self.clear()
		self.draw()

	def draw(self):
		# drawing grid
		for i in range(self.game.width // self.game.tilesize):
			pyglet.graphics.draw_indexed(2, pyglet.gl.GL_LINES,
				[0, 1],
				('v2i', (self.game.offset + i * self.game.tilesize, self.game.offset,
						self.game.offset + i * self.game.tilesize, self.game.height - self.game.offset)),
				('c4B', (255, 255, 255, 80, \
						255, 255, 255, 80))
			)

		for i in range(self.game.height // self.game.tilesize):
			pyglet.graphics.draw_indexed(2, pyglet.gl.GL_LINES,
				[0, 1],
				('v2i', (self.game.offset, self.game.offset + i * self.game.tilesize,
						self.game.width - self.game.offset, self.game.offset + i * self.game.tilesize)),
				('c4B', (255, 255, 255, 80, \
						255, 255, 255, 80))
				)

		#drawing snake
		for part in self.game.snake.parts:
			self.tile.blit(self.game.offset + part[0] * self.game.tilesize + 1, \
							self.game.offset + part[1] * self.game.tilesize + 1)

		#drawing food
		self.apple.blit(self.game.offset + self.game.food.position[0] * self.game.tilesize + 1, \
						self.game.offset + self.game.food.position[1] * self.game.tilesize + 1)

	def update(self, dt):
		self.game.do_move(self.game.choose_action())