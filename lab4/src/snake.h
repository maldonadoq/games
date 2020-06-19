#pragma once

#include <deque>
#include "utils.h"

using std::deque;

/*
Snake direc:
	0: Right    1: Left
	2: Up       3: Down
*/

class Snake{
public:
	deque<Point> body;
	Point head;
	int dim;

	Snake();
	Snake(int);
	
	void grow(Point);
	void move(int);
	bool collision(Point);
	void reset();
	void getData(Point &, Data &);

	~Snake();
};

Snake::Snake(){
}

Snake::Snake(int dim){
	this->dim = dim;
	std::cout << this->dim << std::endl;
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

void Snake::getData(Point & apple, Data &input){		
	Point snake_dir = this->head - this->body[1];
	Point apple_dir = apple - this->head;

	input.front = collision(this->head + snake_dir);
	input.right = collision(this->head + Point({snake_dir.y, -snake_dir.x}));
	input.left  = collision(this->head + Point({-snake_dir.y, snake_dir.x}));	

	float tmpX, tmpY;

	direction(snake_dir, tmpX, tmpY);
	input.snake_x = tmpX;
	input.snake_y = tmpY;

	direction(apple_dir, tmpX, tmpY);
	input.apple_x = tmpX;
	input.apple_y = tmpY;
}

Snake::~Snake(){
	this->body.clear();
}