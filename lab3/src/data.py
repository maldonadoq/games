from .snake import *
from tqdm import tqdm

class SnakeData:
	def __init__(self):		
		self.train_x = []
		self.train_y = []

	def generate(self, games, steps):
		for _ in tqdm(range(games)):
			snake_start, snake_pos, apple_pos, score = starting_position()
			apple_dist = apple_dist_from_snake(apple_pos, snake_pos)

			for _ in range(steps):
				angle, snake_dir_vect, apple_dir_vect_norm, snake_dir_vect_norm = angle_with_apple(snake_pos, apple_pos)

				direction, button_direction = generate_random_direction(snake_pos, angle)
				curr_dir_vect, front_block, left_block, right_block = block_directions(snake_pos)

				direction, button_direction = self.generate_training_data_y(snake_pos, angle,
																			button_direction, direction,
																			front_block, left_block, right_block)					

				if(front_block == 1 and left_block == 1 and right_block == 1):
					break
			
				self.train_x.append([left_block, front_block, right_block, apple_dir_vect_norm[0],
									snake_dir_vect_norm[0], apple_dir_vect_norm[1], snake_dir_vect_norm[1]])

				snake_pos, apple_pos, score = play(snake_start, snake_pos, apple_pos,
															button_direction, score)

		return self.train_x, self.train_y

	def generate_training_data_y(self, snake_pos, angle, button_direction, direction, front_block, left_block, right_block):
		if(direction == -1):
			if(left_block == 1):
				if(front_block == 1 and right_block == 0):
					direction, button_direction = direction_vector(snake_pos, angle, 1)
					self.train_y.append([0,0,1])
				elif(front_block == 0 and right_block == 1):
					direction, button_direction = direction_vector(snake_pos, angle, 0)
					self.train_y.append([0,1,0])
				elif(front_block == 0 and right_block == 0):
					direction, button_direction = direction_vector(snake_pos, angle, 1)
					self.train_y.append([0,0,1])
			else:
				self.train_y.append([1,0,0])
		
		elif(direction == 0):
			if(front_block == 1):
				if(left_block == 1 and right_block == 0):
					direction, button_direction = direction_vector(snake_pos, angle, 1)
					self.train_y.append([0,0,1])
				elif(left_block == 0 and right_block == 1):
					direction, button_direction = direction_vector(snake_pos, angle, -1)
					self.train_y.append([1,0,0])
				elif(left_block == 0 and right_block == 0):
					direction, button_direction = direction_vector(snake_pos, angle, 1)
					self.train_y.append([0,0,1])
			else:
				self.train_y.append([0,1,0])
		
		else:
			if(right_block == 1):
				if(left_block == 1 and front_block == 0):
					direction, button_direction = direction_vector(snake_pos, angle, 0)
					self.train_y.append([0,1,0])
				elif(left_block == 0 and front_block == 1):
					direction, button_direction = direction_vector(snake_pos, angle, -1)
					self.train_y.append([1,0,0])
				elif(left_block == 0 and front_block == 0):
					direction, button_direction = direction_vector(snake_pos, angle, -1)
					self.train_y.append([1,0,0])
			else:
				self.train_y.append([0,0,1])

		return direction, button_direction
		
	def save(self):
		trainXFile = open('data/trainX.csv', 'w')
		trainXFile.write('Left,Front,Right,ADirX,ADirY,SDirX,SDirY\n')
		for x in self.train_x:
			tmp = ''
			for i, v in enumerate(x):
				if(i == len(x)-1):
					tmp += str(v) + '\n'
				else:
					tmp += str(v) + ','
			trainXFile.write(tmp)

		trainYFile = open('data/trainY.csv', 'w')
		trainYFile.write('Left,Same,Right\n')
		for y in self.train_y:
			tmp = ''
			for i, v in enumerate(y):
				if(i == len(y)-1):
					tmp += str(v) + '\n'
				else:
					tmp += str(v) + ','
			trainYFile.write(tmp)
