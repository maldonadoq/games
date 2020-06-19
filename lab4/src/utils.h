#pragma once

#include <math.h>

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

float norm(Point p){
    float tmp = std::sqrt((p.x*p.x) + (p.y*p.y));
    return (tmp == 0)? 1: tmp;
}

void direction(Point p, float &tmpX, float &tmpY){
    float n = norm(p);
    tmpX = (float)p.x / n;
    tmpY = (float)p.y / n;
}

struct Data{
	int left;
	int front;
	int right;
	float apple_x;
	float apple_y;
	float snake_x;
	float snake_y;
};

std::ostream& operator<<(std::ostream& out, const Data& d){
	out << "[" << d.front << "," << d.left << "," << d.right << "," << d.snake_x << "," << d.snake_y << "," << d.apple_x << "," << d.apple_y << "]";
	return out;
}

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