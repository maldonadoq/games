import pygame as pg
from tqdm import tqdm
from src.snake import *
import warnings

with warnings.catch_warnings():  
	warnings.filterwarnings("ignore",category=FutureWarning)
	from keras.models import load_model

width, height = 500, 500

if __name__ == "__main__":
	pg.init()
	display = pg.display.set_mode((width, height))
	pg.display.set_caption('Snake Neural Network')

	model = load_model('data/model.h5')

	games = 2
	steps = 1000

	max_score = 2
	avg_score = 0

	for _ in tqdm(range(games)):
		snake_start, snake_pos, apple_pos, score = starting_position()

		count_same_direction = 0
		prev_direction = 0		

		for _ in range(steps):
			curr_dir_vect, front_block, left_block, right_block = block_directions(snake_pos)

			angle, snake_dir_vect, apple_dir_vect_norm, snake_dir_vect_norm = angle_between(snake_pos, apple_pos)
			
			pred = model.predict(np.array([	left_block, front_block, right_block,
											apple_dir_vect_norm[0], snake_dir_vect_norm[0], apple_dir_vect_norm[1],
											snake_dir_vect_norm[1]]).reshape(-1, 7))
											
			pred_dir = np.argmax(np.array(pred)) - 1

			if(pred_dir == prev_direction):
				count_same_direction += 1
			else:
				count_same_direction = 0
				prev_direction = pred_dir

			new_direction = np.array(snake_pos[0]) - np.array(snake_pos[1])
			if(pred_dir == -1):
				new_direction = np.array([new_direction[1], -new_direction[0]])

			if(pred_dir == 1):
				new_direction = np.array([-new_direction[1], new_direction[0]])

			btn_dir = gen_btn_dir(new_direction)

			next_step = snake_pos[0] + curr_dir_vect
			if(collision_with_boundaries(snake_pos[0]) == 1 or collision_with_self(next_step.tolist(), snake_pos) == 1):
				break
			
			snake_pos, apple_pos, score = play_game(snake_start, snake_pos, apple_pos, btn_dir, score, display)

			if(score > max_score):
				max_score = score

		avg_score += score

	pg.quit()

	print('Max Score:  ', max_score)
	print('Avg score:  ', avg_score/games)
