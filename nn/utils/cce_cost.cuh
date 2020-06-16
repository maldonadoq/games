#pragma once
#include "matrix.cuh"

class CCECost {
public:
	float cost(Matrix predictions, Matrix target);
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
