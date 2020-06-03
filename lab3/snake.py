import pygame as pg
import numpy as np
import random

class Game:
	def starting_position(self):
		snake_start = [100,100]
		snake_pos = [[100,100], [90,100], [80,100]]
		apple_pos = [random.randrange(1,50)*10, random.randrange(1,50)*10]
		score = 3

		return snake_start, snake_pos, apple_pos, score

	def apple_dist_from_snake(self, apple_pos, snake_pos):
		return np.linalg.norm(np.array(apple_pos) - np.array(snake_pos[0]))