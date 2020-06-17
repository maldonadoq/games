#pragma once

#include "../../nn/utils/matrix.cuh"
#include <vector>

void printMatrix(const Matrix&);
std::vector<int> firstResultInt(const Matrix &, int);
std::vector<float> firstResultFloat(const Matrix &, int);

template<typename T>
void printVector(const std::vector<T> &);