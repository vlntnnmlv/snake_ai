from torch.optim import Adam
from random import randint, choice
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pyglet.window import key
from app import *

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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DQNAgent(nn.Module):
	def __init__(self):
		super().__init__()
		self.reward = 0
		self.gamma = 0.9
		self.dataframe = pd.DataFrame()
		self.short_memory = np.array([])
		self.agent_target = 1
		self.agent_predict = 0.0005
		self.epsilon = 0.01
		self.actual = []

		self.first_layer = 30
		self.second_layer = 120
		self.third_layer = 4
		self.memory = collections.deque(maxlen = 100)
		self.weights = None
		self.load_weights = None
		self.optimizer = None
		self.network()

	def network(self):
		self.f1 = nn.Linear(8, self.first_layer)
		self.f2 = nn.Linear(self.first_layer, self.second_layer)
		self.f3 = nn.Linear(self.second_layer, self.third_layer)
		self.f4 = nn.Linear(self.third_layer, 4)

	def forward(self, X):
		x = F.relu(self.f1(X))
		x = F.relu(self.f2(x))
		x = F.relu(self.f3(x))
		x = F.softmax(self.f4(x), dim = -1)
		return x

	def get_state(self, game):
		s = [
			game.snake.direction[0],
			game.snake.direction[1],
			int(game.snake.parts[-1][1] < game.food.position[1]),
			int(game.snake.parts[-1][1] > game.food.position[1]),
			int(game.snake.parts[-1][0] < game.food.position[0]),
			int(game.snake.parts[-1][0] > game.food.position[0]),
			int(game.snake.parts[-1][0] == 0 or game.snake.parts[-1][0] == game.tilew - 1),
			int(game.snake.parts[-1][1] == 0 or game.snake.parts[-1][1] == game.tileh - 1),
		]

		return np.asarray(s)

	def set_reward(self, crash, fed):
		self.reward = 0
		if crash:
			self.reward = -10
		if fed:
			self.reward = 10
		return self.reward

	def remember(self, state, action, next_state, reward, done):
		self.memory.append((state, action, reward, next_state, done))

	def replay_memory(self, memory, batch_size):
		if (len(memory) > batch_size):
			minibatch = random.sample(memory, batch_size)
		else:
			minibatch = memory
		for state, action, next_state, reward, done in minibatch:
			self.train()
			torch.set_grad_enabled(True)
			target = reward
			next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype = torch.float32).to(DEVICE)
			state_tensor = torch.tensor(np.expand_dims(state, 0), dtype = torch.float32, requires_grad = True).to(DEVICE)
			if not done:
				target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
			output = self.forward(state_tensor)
			target_f = output.clone()
			target_f[0][np.argmax(action)] = targett
			target_f.detach()
			self.optimizer.zero_grad()
			loss = F.MSELoss(output, target_f)
			loss.backward()
			self.optimizer.step()

	def train_short_memory(self, state, action, reward, next_state, done):
		self.train()
		torch.set_grad_enabled(True)
		target = reward
		next_state_tensor = torch.tensor(next_state.reshape((1, 8)), dtype=torch.float32).to(DEVICE)
		state_tensor = torch.tensor(state.reshape((1, 8)), dtype=torch.float32, requires_grad=True).to(DEVICE)
		if not done:
			target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
		output = self.forward(state_tensor)
		target_f = output.clone()
		target_f[0][np.argmax(action)] = target
		target_f.detach()
		self.optimizer.zero_grad()
		loss = F.mse_loss(output, target_f)
		loss.backward()
		self.optimizer.step()

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
			return True
		return False

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

		self.agent = DQNAgent()
		self.agent = self.agent.to(DEVICE)
		self.agent.optimizer = optim.Adam(self.agent.parameters(), weight_decay = 0, lr = 0.0005)
		self.counter_games = 0
		self.score_plot = []
		self.counter_plot = []
		self.record = 0
		self.total_score = 0

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
		pyglet.clock.schedule_interval(self.update, 1 / 120)

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
		steps = 0


		state_old = self.agent.get_state(self)
		if random.uniform(0,1) < self.agent.epsilon:
			final_move = np.eye(4)[randint(0,3)]
		else:
			with torch.no_grad():
				state_old_tensor = torch.tensor(state_old.reshape((1, 8)), dtype=torch.float32).to(DEVICE)
				prediction = self.agent(state_old_tensor)
				final_move = np.eye(4)[np.argmax(prediction.detach().cpu().numpy()[0])]

		f = np.argmax(final_move)
		if f == 0:
			self.snake.turn((0, 1))
		if f == 1:
			self.snake.turn((0, -1))
		if f == 2:
			self.snake.turn((1, 0))
		if f == 3:
			self.snake.turn((-1, 0))

		# MOVE HAPPENS HERE
		fed = False
		crash = self.snake.move()
		if (self.snake.parts[-1] == self.food.position):
			fed = True
			self.food.reset()
			self.snake.grow(self.food)
		# MOVE ENDS HERE

		state_new = self.agent.get_state(self)
		reward = self.agent.set_reward(crash, fed)

		if reward > 0:
			steps = 0

		self.agent.train_short_memory(state_old, final_move, reward, state_new, crash)
		self.agent.remember(state_old, final_move, reward, state_new, crash)
		steps += 1

		if crash:
			steps = 0
			self.snake.reset()