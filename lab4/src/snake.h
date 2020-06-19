#pragma once

#include <iostream>
#include <deque>
#include <vector>

#include "utils.h"

using std::vector;
using std::deque;
using std::cout;
using std::endl;

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
	bool play(const vector<int> &);

	vector<float> getData(Point &);

	~Snake();
};