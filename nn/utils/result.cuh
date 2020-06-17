#pragma once

#include "matrix.cuh"
#include <vector>

void printMatrix(const Matrix&);
int computeAccuracy(const Matrix &, const Matrix &, int);
std::vector<float> firstResult(const Matrix &, int);
void printVector(const std::vector<float> &);