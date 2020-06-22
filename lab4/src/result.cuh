#pragma once

#include "../../nn/utils/tensor.cuh"
#include <vector>

void printMatrix(const Tensor&);
std::vector<int> firstResultInt(const Tensor &, int);
std::vector<float> firstResultFloat(const Tensor &, int);

template<typename T>
void printVector(const std::vector<T> &);