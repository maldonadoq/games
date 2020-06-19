#pragma once

#include <deque>

using std::deque;

struct Point{
	int x;
	int y;

	Point operator-(const Point rhs){ 
		return {x-rhs.x, y-rhs.y};
	}

	Point operator+(const Point rhs){ 
		return {x+rhs.x, y+rhs.y};
	}
};

std::ostream& operator<<(std::ostream& out, const Point& p){
	out << "[" << p.x << "," << p.y  << "]";
	return out;
}

struct Data{
	float left;
	float front;
	float right;
	float apple_x;
	float apple_y;
	float snake_x;
	float snake_y;
};

/*
Snake direc:
	0: Right    1: Left
	2: Up       3: Down
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

	Snake();
	Snake(int);
	
	void grow(Point);
	void move(int);
	bool isInBody(Point);
	void reset();
	void getData(Data &);

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

bool Snake::isInBody(Point point){
	for(auto p: this->body){
		if((p.x == point.x) and (p.y == point.y)){
			return true;
		}
	}

	return false;
}

void Snake::getData(Data &input){		
	Point dir = this->head - this->body[1];

	input.front = isInBody(this->head + dir);
	input.right = isInBody(this->head + Point({dir.y, -dir.x}));
	input.left  = isInBody(this->head + Point({-dir.y, dir.x}));
}

Snake::~Snake(){
	this->body.clear();
}
