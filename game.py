from random import randint, choice
from DQN import *

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

class Game():
	def __init__(self, width = 510, height = 510, offset = 5, tilesize = 10):
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
		self.agent.optimizer = optim.Adam(self.agent.parameters(), weight_decay = 0, lr = 0.0005)
		self.counter_games = 0
		self.score_plot = []
		self.counter_plot = []
		self.record = 0
		self.total_score = 0

		self.snake = Snake(self.tilew, self.tileh)
		self.food = Food(self.tilew, self.tileh)

	def do_move(self, symbol):
		if symbol == 'U':
			self.snake.turn((0, 1))
		if symbol == 'D':
			self.snake.turn((0, -1))
		if symbol == 'L':
			self.snake.turn((-1, 0))
		if symbol == 'R':
			self.snake.turn((1, 0))
 
	def update(self):
		
		# print(self.steps, self.score)

		# MOVE HAPPENS HERE
		self.fed = False
		self.crash = self.snake.move()
		if (self.snake.parts[-1] == self.food.position):
			self.fed = True
			self.food.reset()
			self.score += 10
			self.snake.grow(self.food)
		# MOVE ENDS HERE

		self.steps += 1
		
	def run(self):
		print("Let the game begin!")
		while self.games_counter <= 100:
			self.steps = 0
			self.score = 0
			self.crash = False
			self.snake.reset()
			while not self.crash and self.steps < 100:
				self.agent.epsilon = 1 - self.games_counter * 0.0005

				state_old = self.agent.get_state(self)
				if random.uniform(0,1) < self.agent.epsilon:
					final_move = np.eye(4)[randint(0,3)]
				else:
					with torch.no_grad():
						state_old_tensor = torch.tensor(state_old.reshape((1, 8)), dtype=torch.float32).to(DEVICE)
						prediction = self.agent(state_old_tensor)
						final_move = np.eye(4)[np.argmax(prediction.detach().cpu().numpy()[0])]

				self.update()
				if self.fed:
					self.score += 10

				state_new = self.agent.get_state(self)
				reward = self.agent.set_reward(self.crash, self.fed)

				if reward > 0:
					self.steps = 0

				self.agent.train_short_memory(state_old, final_move, reward, state_new, self.crash)
				self.agent.remember(state_old, final_move, reward, state_new, self.crash)
			self.games_counter += 1
			if self.score > self.record:
				self.record = self.score
		print("The game ended with record of", self.record)