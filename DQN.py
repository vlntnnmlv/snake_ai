from torch.optim import Adam
from random import randint, choice
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pyglet.window import key
from math import sqrt

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
	def __init__(self, inp = 10, outp = 3, l1 = 64, l2 = 64, l3 = 64):
		super().__init__()
		self.reward = 0
		self.gamma = 0.9
		self.dataframe = pd.DataFrame()
		self.short_memory = np.array([])
		self.agent_target = 1
		self.agent_predict = 0.0005
		self.epsilon = 0.01
		self.actual = []

		self.inp = inp
		self.first_layer = l1
		self.second_layer = l2
		self.third_layer = l3
		self.outp = outp
		self.memory = collections.deque(maxlen = 100_000)
		self.weights = None
		self.load_weights = None
		self.optimizer = None
		self.network()

	def network(self):
		self.f1 = nn.Linear(self.inp, 256)
		# self.f2 = nn.Linear(self.first_layer, self.second_layer)
		# self.f3 = nn.Linear(self.second_layer, self.third_layer)
		# self.f2 = nn.Linear(self.first_layer, self.third_layer)
		self.f4 = nn.Linear(256, self.outp)

	def forward(self, x):
		x = F.relu(self.f1(x))
		# x = F.relu(self.f2(x))
		# x = F.relu(self.f3(x))
		x = F.softmax(self.f4(x), dim = -1)
		return x

	def get_state(self, game):
		s = [
			int(game.snake.direction == (0, 1)),
			int(game.snake.direction == (0, -1)),
			int(game.snake.direction == (1, 0)),
			int(game.snake.direction == (-1, 0)),
			int(game.snake.parts[-1][1] < game.food.position[1]),
			int(game.snake.parts[-1][1] > game.food.position[1]),
			int(game.snake.parts[-1][0] < game.food.position[0]),
			int(game.snake.parts[-1][0] > game.food.position[0]),
			int(game.snake.parts[-1][0] == 0 or game.snake.parts[-1][0] == game.tilew - 1),
			int(game.snake.parts[-1][1] == 0 or game.snake.parts[-1][1] == game.tileh - 1),
		]

		return np.asarray(s)

	def set_reward(self, crash, fed, steps, apples):
		if crash:
			self.reward = -10
		if crash and steps < 15:
			self.reward = -50
		if fed:
			self.reward = sqrt(apples) * 3.5
		if not fed and not crash:
			self.reward = -0.5
		return self.reward

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def replay_new(self, memory, batch_size):
		if (len(memory) > batch_size):
			minibatch = random.sample(memory, batch_size)
		else:
			minibatch = memory
		for state, action, reward, next_state, done in minibatch:
			self.train()
			torch.set_grad_enabled(True)
			target = reward
			next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype = torch.float32).to(DEVICE)
			state_tensor = torch.tensor(np.expand_dims(state, 0), dtype = torch.float32, requires_grad = True).to(DEVICE)
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

	def train_short_memory(self, state, action, reward, next_state, done):
		self.train()
		torch.set_grad_enabled(True)
		target = reward
		next_state_tensor = torch.tensor(next_state.reshape((1, self.inp)), dtype=torch.float32).to(DEVICE)
		state_tensor = torch.tensor(state.reshape((1, self.inp)), dtype=torch.float32, requires_grad=True).to(DEVICE)
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
		return loss