import numpy as np
import random
import math

dimention = 500

def starting_position():
	snake_start = [100,100]
	snake_pos = [[100,100], [90,100], [80,100]]
	apple_pos = [random.randrange(1,50)*10, random.randrange(1,50)*10]
	score = 3

	return snake_start, snake_pos, apple_pos, score

def apple_dist_from_snake(apple_pos, snake_pos):
	return np.linalg.norm(np.array(apple_pos) - np.array(snake_pos[0]))

def angle_with_apple(snake_pos, apple_pos):
	apple_dir_vect = np.array(apple_pos) - np.array(snake_pos[0])
	snake_dir_vect = np.array(snake_pos[0]) - np.array(snake_pos[1])

	norm_of_apple_dir_vect = np.linalg.norm(apple_dir_vect)
	norm_of_snake_dir_vect = np.linalg.norm(snake_dir_vect)
	if(norm_of_apple_dir_vect == 0):
		norm_of_apple_dir_vect = 10
	if(norm_of_snake_dir_vect == 0):
		norm_of_snake_dir_vect = 10

	apple_dir_vect_norm = apple_dir_vect / norm_of_apple_dir_vect
	snake_dir_vect_norm = snake_dir_vect / norm_of_snake_dir_vect
	angle = math.atan2(
			apple_dir_vect_norm[1] * snake_dir_vect_norm[0] - apple_dir_vect_norm[0] * snake_dir_vect_norm[1],
			apple_dir_vect_norm[1] * snake_dir_vect_norm[1] + apple_dir_vect_norm[0] * snake_dir_vect_norm[0]
		) / math.pi
	return angle, snake_dir_vect, apple_dir_vect_norm, snake_dir_vect_norm

def generate_random_direction(snake_pos, angle_with_apple):
	direction = 0
	if(angle_with_apple > 0):
		direction = 1
	elif(angle_with_apple < 0):
		direction = -1
	else:
		direction = 0

	return direction_vector(snake_pos, angle_with_apple, direction)


def direction_vector(snake_pos, angle_with_apple, direction):
	curr_dir_vect = np.array(snake_pos[0]) - np.array(snake_pos[1])
	left_dir_vect = np.array([curr_dir_vect[1], -curr_dir_vect[0]])
	right_dir_vect = np.array([-curr_dir_vect[1], curr_dir_vect[0]])

	new_direction = curr_dir_vect

	if(direction == -1):
		new_direction = left_dir_vect
	if(direction == 1):
		new_direction = right_dir_vect

	button_direction = generate_button_direction(new_direction)

	return direction, button_direction


def generate_button_direction(new_direction):
	button_direction = 0
	if(new_direction.tolist() == [10, 0]):
		button_direction = 1
	elif(new_direction.tolist() == [-10, 0]):
		button_direction = 0
	elif(new_direction.tolist() == [0, 10]):
		button_direction = 2
	else:
		button_direction = 3

	return button_direction

def block_directions(snake_pos):
	curr_dir_vect = np.array(snake_pos[0]) - np.array(snake_pos[1])

	left_dir_vect = np.array([curr_dir_vect[1], -curr_dir_vect[0]])
	right_dir_vect = np.array([-curr_dir_vect[1], curr_dir_vect[0]])

	front_block = direction_block(snake_pos, curr_dir_vect)
	left_block = direction_block(snake_pos, left_dir_vect)
	right_block = direction_block(snake_pos, right_dir_vect)

	return curr_dir_vect, front_block, left_block, right_block


def direction_block(snake_pos, curr_dir_vect):
	next_step = snake_pos[0] + curr_dir_vect
	snake_start = snake_pos[0]
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

def generate_snake(snake_start, snake_pos, apple_pos, button_direction, score):
	if button_direction == 1:
		snake_start[0] += 10
	elif button_direction == 0:
		snake_start[0] -= 10
	elif button_direction == 2:
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

def play(snake_start, snake_pos, apple_pos, button_direction, score):	
	snake_pos, apple_pos, score = generate_snake(snake_start, snake_pos, apple_pos, button_direction, score)
	return snake_pos, apple_pos, score