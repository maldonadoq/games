#pragma once
#include "matrix.cuh"

class CCELoss {
public:
	float cost(Matrix predictions, Matrix target);
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
