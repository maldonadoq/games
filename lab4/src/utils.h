#pragma once

#include <math.h>
#include <iostream>
#include <vector>

struct Point{
	int x;
	int y;

	Point operator-(const Point rhs){ 
		return {x-rhs.x, y-rhs.y};
	}

	Point operator+(const Point rhs){ 
		return {x+rhs.x, y+rhs.y};
	}

	bool operator==(const Point rhs){ 
		return (x == rhs.x) and (y == rhs.y);
	}
};

struct Data{
	int left;
	int front;
	int right;
	float apple_x;
	float apple_y;
	float snake_x;
	float snake_y;
};

std::ostream& operator<<(std::ostream& out, const Data&);
std::ostream& operator<<(std::ostream& out, const Point&);

int mod(int, int);
float norm(Point);
void direction(Point, float &, float &);
int argmax(const std::vector<int> &);