from random import randint, choice
from DQN import *

import matplotlib.pyplot as plt

class Snake:
	def __init__(self, tilew, tileh):
		self.tilew = tilew
		self.tileh = tileh
		self.parts = []
		self.reset()
		self.direction = choice([(0,1),(0,-1),(1,0),(-1,0)])
		self.apples_eaten = 0

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

class Game():
	def __init__(self, width = 410, height = 410, offset = 5, tilesize = 20):
		self.score = 0
		self.games_counter = 0
		self.crash = False
		self.fed = False
		self.steps = 0
		self.record = 0

		self.width = width
		self.height = height
		self.offset = offset
		self.tilesize = tilesize
		self.tilew = (self.width - self.offset * 2) // self.tilesize
		self.tileh = (self.height - self.offset * 2) // self.tilesize

		self.agent = DQNAgent()
		self.agent = self.agent.to(DEVICE)
		self.agent.optimizer = optim.Adam(self.agent.parameters(), weight_decay = 0, lr = 0.001)
		self.score_plot = []
		self.counter_plot = []
		self.total_score = 0

		self.snake = Snake(self.tilew, self.tileh)
		self.food = Food(self.tilew, self.tileh)

	def update(self):
		
		# MOVE HAPPENS HERE
		self.check_on_crash()
		self.check_on_fed()
		if self.crash:
			self.snake.reset()
			self.snake.apples_eaten = 0
			self.steps = 0
			self.crash = False
		if (self.fed):
			self.score += 1
			self.snake.apples_eaten += 1
			self.snake.grow(self.food)
			self.food.reset()
		# MOVE ENDS HERE

		self.steps += 1

	def check_on_crash(self):
		if (self.snake[-1][0] < 0 or self.snake[-1][0] >= self.tilew or \
				self.snake[-1][1] < 0 or self.snake[-1][1] >= self.tileh):
			self.crash = True
		else:
			self.crash = False

	def check_on_fed(self):
		if self.snake[-1] == self.food.position:
			self.fed = True
		else:
			self.fed = False

	def do_move(self, action):
		if action == 0:
			self.snake.turn((-self.snake.direction[1], self.snake.direction[0]))
		if action == 1:
			pass
		if action == 2:
			self.snake.turn((self.snake.direction[1], -self.snake.direction[0]))
		self.snake.move()

	def initialize_game(self, batch_size):
		state_init1 = self.agent.get_state(self)
		action = np.array([1, 0, 0, 0])
		self.do_move(np.argmax(action))
		state_init2 = self.agent.get_state(self)
		reward1 = self.agent.set_reward(self.crash, self.fed, self.steps, self.snake.apples_eaten)
		self.agent.remember(state_init1, action, reward1, state_init2, self.crash)
		self.agent.replay_new(self.agent.memory, batch_size)
	
	def choose_action(self):
		state_old = self.agent.get_state(self)
		if random.uniform(0,1) < self.agent.epsilon:
			final_move = np.eye(self.agent.outp)[randint(0,self.agent.outp - 1)]
		else:
			with torch.no_grad():
				state_old_tensor = torch.tensor(state_old.reshape((1, self.agent.inp)), dtype=torch.float32).to(DEVICE)
				prediction = self.agent(state_old_tensor)
				final_move = np.eye(self.agent.outp)[np.argmax(prediction.detach().cpu().numpy()[0])]
		action = np.argmax(final_move)
		return action

	def step(self, train = True):
		if not train:
			self.epsilon = 0.01

		state_old = self.agent.get_state(self)
		if random.uniform(0,1) < self.agent.epsilon:
			final_move = np.eye(self.agent.outp)[randint(0,self.agent.outp - 1)]
		else:
			with torch.no_grad():
				state_old_tensor = torch.tensor(state_old.reshape((1, self.agent.inp)), dtype=torch.float32).to(DEVICE)
				prediction = self.agent(state_old_tensor)
				final_move = np.eye(self.agent.outp)[np.argmax(prediction.detach().cpu().numpy()[0])]
		action = np.argmax(final_move)
		self.do_move(action)

		self.check_on_crash()
		self.check_on_fed()
		if self.fed:
			self.snake.grow(self.food)
			self.food.reset()
			self.score += 1
			self.steps = 0

		state_new = self.agent.get_state(self)
		reward = self.agent.set_reward(self.crash, self.fed, self.steps, self.snake.apples_eaten)
		if train:
			self.agent.train_short_memory(state_old, final_move, reward, state_new, self.crash)
			self.agent.remember(state_old, final_move, reward, state_new, self.crash)

	def run_train(self):
		print("Let the game begin!")
		while self.games_counter <= 100:
			# reset environment
			self.steps = 0
			self.score = 0
			self.crash = False
			self.snake.reset()
			self.food.reset()

			# update epsilon
			self.agent.epsilon = 1 - self.games_counter * 1/110
			
			# game loop
			self.agent.replay_new(self.agent.memory, 1000)
			while not self.crash and self.steps < 100:
				self.step()
			
			self.total_score += self.score
			self.counter_plot.append(self.games_counter)
			self.score_plot.append(self.score)
			self.games_counter += 1
			if self.score > self.record:
				self.record = self.score
			if self.games_counter % 1 == 0:
				print(self.games_counter, "episode, score: ", self.score)
	
		torch.save(self.agent.state_dict(), "./snake_ai")
		plt.scatter(self.counter_plot, self.score_plot)
		plt.show()
		print("The game ended with record of", self.record)