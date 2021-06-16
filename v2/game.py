import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from math import sqrt

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
	RIGHT = 1
	LEFT = 2
	UP = 3
	DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 80

class SnakeGameAI:

	def __init__(self, w = 200, h = 200, tilesize = 20, offset = 5):
		self.w = w
		self.h = h
		self.ts = tilesize
		self.os = offset
		self.tw = self.w // self.ts
		self.th = self.h // self.ts
		# init display
		self.display = pygame.display.set_mode((self.w + self.os * 2, self.h + self.os * 2))
		pygame.display.set_caption('Snake')
		self.clock = pygame.time.Clock()
		self.reset()


	def reset(self):
		# init game state
		self.direction = Direction.RIGHT

		self.head = Point(self.tw / 2, self.th / 2)
		self.snake = [self.head,
					Point(self.head.x - 1, self.head.y),
					Point(self.head.x - 2, self.head.y)]

		self.score = 0
		self.food = None
		self._place_food()
		self.frame_iteration = 0

	def _place_food(self):
		x = random.randint(0, self.tw - 1)
		y = random.randint(0, self.th - 1)
		self.food = Point(x, y)
		if self.food in self.snake:
			self._place_food()


	def play_step(self, action):
		self.frame_iteration += 1
		# 1. collect user input
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
		
		# 2. move
		self._move(action) # update the head
		self.snake.insert(0, self.head)
		
		# 3. check if game over
		reward = -0.5
		game_over = False
		if self.is_collision() or self.frame_iteration > 100*len(self.snake):
			game_over = True
			reward = -10
			return reward, game_over, self.score

		# 4. place new food or just move
		if self.head == self.food:
			self.score += 1
			reward = 3.5 * sqrt(self.score)
			self._place_food()
		else:
			self.snake.pop()
		
		# 5. update ui and clock
		self._update_ui()
		self.clock.tick(SPEED)
		# 6. return game over and score
		return reward, game_over, self.score


	def is_collision(self, pt=None):
		if pt is None:
			pt = self.head
		# hits boundary
		if pt.x >= self.tw or pt.x < 0 or pt.y >= self.th or pt.y < 0:
			return True
		# hits itself
		if pt in self.snake[1:]:
			return True

		return False


	def _update_ui(self):
		self.display.fill(BLACK)

		# draw grid
		for i in range(self.tw + 1):
			pygame.draw.line(self.display, (100, 100, 100), \
				(self.os + i * self.ts, self.os), (self.os + i * self.ts, self.h + self.os))
	
		for i in range(self.th + 1):
			pygame.draw.line(self.display, (100, 100, 100), \
				(self.os, self.os + i * self.ts), (self.w + self.os, self.os + i * self.ts))

		for pt in self.snake:
			pygame.draw.rect(self.display, (0, 0, 255), \
				pygame.Rect(self.os + pt.x * self.ts, self.os + pt.y * self.ts, self.ts, self.ts))

		pygame.draw.rect(self.display, (255, 0, 0), \
			pygame.Rect(self.os + self.food.x * self.ts, self.os + self.food.y * self.ts, self.ts, self.ts))

		text = font.render("Score: " + str(self.score), True, WHITE)
		self.display.blit(text, [0, 0])
		pygame.display.flip()


	def _move(self, action):
		# [straight, right, left]

		clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
		idx = clock_wise.index(self.direction)

		if np.array_equal(action, [1, 0, 0]):
			new_dir = clock_wise[idx] # no change
		elif np.array_equal(action, [0, 1, 0]):
			next_idx = (idx + 1) % 4
			new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
		else: # [0, 0, 1]
			next_idx = (idx - 1) % 4
			new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

		self.direction = new_dir

		x = self.head.x
		y = self.head.y
		if self.direction == Direction.RIGHT:
			x += 1
		elif self.direction == Direction.LEFT:
			x -= 1
		elif self.direction == Direction.DOWN:
			y += 1
		elif self.direction == Direction.UP:
			y -= 1

		self.head = Point(x, y)