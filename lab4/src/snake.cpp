#include "snake.h"

Snake::Snake(){
}

Snake::Snake(int dim){
	this->dim = dim;
	reset();
}

void Snake::grow(Point point){
	this->body.push_back(point);
}

void Snake::reset(){
	this->head = {1,0};

	this->body.clear();
	this->body.push_back({1,0});
	this->body.push_back({0,0});
}

void Snake::move(int dir){
	switch (dir){
		case 0:{
			this->head.x = mod(this->head.x + 1, this->dim);
			this->body.push_front(this->head);
			this->body.pop_back();
			break;
		}
		case 1:{
			this->head.x = mod(this->head.x - 1, this->dim);
			this->body.push_front(this->head);
			this->body.pop_back();
			break;
		}
		case 2:{
			this->head.y = mod(this->head.y + 1, this->dim);
			this->body.push_front(this->head);
			this->body.pop_back();
			break;
		}
		case 3:{
			this->head.y = mod(this->head.y - 1, this->dim);
			this->body.push_front(this->head);
			this->body.pop_back();
			break;
		}
		default:
			break;
	}
}

bool Snake::collision(Point point){
	for(auto p: this->body){
		if((p.x == point.x) and (p.y == point.y)){
			return true;
		}
	}

	return false;
}

/*
pred = model.predict(np.array([	left_block, front_block, right_block,
											apple_dir_vect_norm[0], snake_dir_vect_norm[0], apple_dir_vect_norm[1],
											snake_dir_vect_norm[1]]).reshape(-1, 7))
*/

vector<float> Snake::getData(Point & apple){
	Point snake_dir = this->head - this->body[1];
	Point apple_dir = apple - this->head;

	vector<float> input;
		
	input.push_back(collision(this->head + Point({-snake_dir.y, snake_dir.x})));	// Left
	input.push_back(collision(this->head + snake_dir));								// Front
	input.push_back(collision(this->head + Point({snake_dir.y, -snake_dir.x})));	// Right

	float tmpSx, tmpSy;
	float tmpAx, tmpAy;

	direction(snake_dir, tmpSx, tmpSy);
	direction(apple_dir, tmpAx, tmpAy);
	
	input.push_back(tmpAx);
	input.push_back(tmpSx);
	input.push_back(tmpAy);
	input.push_back(tmpSy);

	return input;
}

Snake::~Snake(){
	this->body.clear();
}