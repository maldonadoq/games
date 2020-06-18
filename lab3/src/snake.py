import numpy as np
import pygame as pg
import random
import math
import time

dimention = 500

def starting_position():
	snake_start = [100,100]
	snake_pos = [[100,100], [90,100], [80,100]]
	apple_pos = [random.randrange(1,50)*10, random.randrange(1,50)*10]
	score = 3

	return snake_start, snake_pos, apple_pos, score

def apple_dist_from_snake(apple_pos, snake_pos):
	return np.linalg.norm(np.array(apple_pos) - np.array(snake_pos[0]))

def angle_between(snake_pos, apple_pos):
	apple_dir = np.array(apple_pos) - np.array(snake_pos[0])
	snake_dir = np.array(snake_pos[0]) - np.array(snake_pos[1])

	norm_apple_dir = np.linalg.norm(apple_dir)
	norm_snake_dir = np.linalg.norm(snake_dir)
	if(norm_apple_dir == 0):
		norm_apple_dir = 10
	if(norm_snake_dir == 0):
		norm_snake_dir = 10

	apple_dir = apple_dir / norm_apple_dir
	snake_dir = snake_dir / norm_snake_dir
	angle = math.atan2(
			apple_dir[1] * snake_dir[0] - apple_dir[0] * snake_dir[1],
			apple_dir[1] * snake_dir[1] + apple_dir[0] * snake_dir[0]
		) / math.pi
	return angle, snake_dir, apple_dir, snake_dir

def gen_rnd_dir(snake_pos, angle_between):
	direc = 0
	if(angle_between > 0):
		direc = 1
	elif(angle_between < 0):
		direc = -1
	else:
		direc = 0

	return dir_vector(snake_pos, angle_between, direc)


def dir_vector(snake_pos, angle_between, direc):
	curr_dir_vect = np.array(snake_pos[0]) - np.array(snake_pos[1])
	left_dir_vect = np.array([curr_dir_vect[1], -curr_dir_vect[0]])
	right_dir_vect = np.array([-curr_dir_vect[1], curr_dir_vect[0]])

	new_dir = curr_dir_vect

	if(direc == -1):
		new_dir = left_dir_vect
	if(direc == 1):
		new_dir = right_dir_vect

	btn_dir = gen_btn_dir(new_dir)

	return direc, btn_dir


def gen_btn_dir(new_dir):
	btn_dir = 0
	if(new_dir.tolist() == [10, 0]):
		btn_dir = 1
	elif(new_dir.tolist() == [-10, 0]):
		btn_dir = 0
	elif(new_dir.tolist() == [0, 10]):
		btn_dir = 2
	else:
		btn_dir = 3

	return btn_dir

def block_directions(snake_pos):
	curr_dir_vect = np.array(snake_pos[0]) - np.array(snake_pos[1])

	left_dir_vect = np.array([curr_dir_vect[1], -curr_dir_vect[0]])
	right_dir_vect = np.array([-curr_dir_vect[1], curr_dir_vect[0]])

	front_block = dir_block(snake_pos, curr_dir_vect)
	left_block = dir_block(snake_pos, left_dir_vect)
	right_block = dir_block(snake_pos, right_dir_vect)

	return curr_dir_vect, front_block, left_block, right_block


def dir_block(snake_pos, curr_dir_vect):
	next_step = snake_pos[0] + curr_dir_vect
	if(collision_with_boundaries(next_step) == 1 or collision_with_self(next_step.tolist(), snake_pos) == 1):
		return 1
	else:
		return 0

def collision_with_apple(apple_pos, score):
	apple_pos = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
	score += 1
	return apple_pos, score


def collision_with_boundaries(snake_start):
	if snake_start[0] >= dimention or snake_start[0] < 0 or snake_start[1] >= dimention or snake_start[1] < 0:
		return 1
	else:
		return 0


def collision_with_self(snake_start, snake_pos):
	# snake_start = snake_pos[0]
	if snake_start in snake_pos[1:]:
		return 1
	else:
		return 0

def generate_snake(snake_start, snake_pos, apple_pos, btn_dir, score):
	if btn_dir == 1:
		snake_start[0] += 10
	elif btn_dir == 0:
		snake_start[0] -= 10
	elif btn_dir == 2:
		snake_start[1] += 10
	else:
		snake_start[1] -= 10

	if snake_start == apple_pos:
		apple_pos, score = collision_with_apple(apple_pos, score)
		snake_pos.insert(0, list(snake_start))

	else:
		snake_pos.insert(0, list(snake_start))
		snake_pos.pop()

	return snake_pos, apple_pos, score

def play(snake_start, snake_pos, apple_pos, btn_dir, score):	
	snake_pos, apple_pos, score = generate_snake(snake_start, snake_pos, apple_pos, btn_dir, score)
	return snake_pos, apple_pos, score

def display_snake(snake_pos, display):
	for pos in snake_pos:
		pg.draw.rect(display, (255, 0, 0), pg.Rect(pos[0], pos[1], 10, 10))

def display_apple(apple_pos, display):
	pg.draw.rect(display, (0, 255, 0), pg.Rect(apple_pos[0], apple_pos[1], 10, 10))

def play_game(snake_start, snake_pos, apple_pos, btn_dir, score, display):
	display.fill((255, 255, 255))

	display_apple(apple_pos, display)
	display_snake(snake_pos, display)

	snake_pos, apple_pos, score = generate_snake(snake_start, snake_pos, apple_pos, btn_dir, score)
	pg.display.update()
	time.sleep(0.025)

	return snake_pos, apple_pos, score