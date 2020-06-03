from snake import Game

class SnakeData:
	def __init__(self, game):
		self.game = game
		self.train_x = []
		self.train_y = []

	def generate(self, games, steps):
		for _ in range(games):
			snake_start, snake_pos, apple_pos, score = self.game.starting_position()
			apple_dist = self.game.apple_dist_from_snake(apple_pos, snake_pos)

			for _ in range(steps):
				angle, snake_dir_vect, apple_dir_vect_norm, snake_dir_vect_norm = self.game.angle_with_apple(snake_pos, apple_pos)

				direction, button_direction = self.game.generate_random_direction(snake_pos, angle)
				curr_dir_vect, front_blocked, left_blocked, right_blocked = self.game.blocked_directions(snake_pos)

				
