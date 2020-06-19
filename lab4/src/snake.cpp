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
	bool self = false;
	bool wall = (point.x == -1) or (point.x == dim) or (point.y == -1) or (point.y == dim);

	for(auto p: this->body){
		if((p.x == point.x) and (p.y == point.y)){
			self = true;
			break;
		}
	}	

	return (wall or self);
}

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

bool Snake::play(const vector<int> &vect){
	int dirP = argmax(vect) - 1;
	Point new_dir = this->head - this->body[1];
	Point tmp;

	if(dirP == 1){
        // Left
		tmp = this->head + Point({-new_dir.y, new_dir.x});

		if(collision(tmp))
			return true;

		this->head = tmp;
		this->body.push_front(this->head);
		this->body.pop_back();
	}
	else if(dirP == -1){
		// Right		
		tmp = this->head + Point({new_dir.y, -new_dir.x});

		if(collision(tmp))
			return true;

		this->head = tmp;
		this->body.push_front(this->head);
		this->body.pop_back();
	}
	else{
		// Front
		tmp = this->head + new_dir;

		if(collision(tmp))
			return true;

		this->head = tmp;
		this->body.push_front(this->head);
		this->body.pop_back();
	}

	return false;
}

Snake::~Snake(){
	this->body.clear();
}