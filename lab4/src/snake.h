#pragma once

#include <deque>

using std::deque;

struct Point{
	int x;
	int y;
};

/*
Snake direc:
	-1: Right    1: Left
	-2: Up       2: Down
*/

int mod(int x, int m){
	int t;
	if(x == -1){
		t = m-1;
	}
	else{
		t = x % m;
	}

	return t;
}

class Snake{
public:
	deque<Point> body;
	Point head;

	int dim;
	int dir;

	Snake();
	Snake(Point, int, int);
	
	void grow(Point);
	void update();
	bool setDir(int);
	void reset();

	~Snake();
};

Snake::Snake(){
}

Snake::Snake(Point point, int dim, int dir=-1){
	this->head = point;
	this->body.push_front(point);

	this->dim = dim;
	this->dir = dir;
}

void Snake::grow(Point point){
	this->body.push_back(point);
}

void Snake::reset(){
	this->head.x = 0;
	this->head.y = 0;

	this->body.clear();
	this->body.push_front(this->head);
}

bool Snake::setDir(int dir){
	if(this->dir == -dir){
		return true;
	}

	this->dir = dir;
	return false;
}

void Snake::update(){

	switch (this->dir){
		case -1:{
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
		case -2:{
			this->head.y = mod(this->head.y + 1, this->dim);
			this->body.push_front(this->head);
			this->body.pop_back();
			break;
		}
		case 2:{
			this->head.y = mod(this->head.y - 1, this->dim);
			this->body.push_front(this->head);
			this->body.pop_back();
			break;
		}
		default:
			break;
	}
}

Snake::~Snake(){
	this->body.clear();
}
