#pragma once

#include "../utils/tensor.cuh"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using std::string;

class SnakeDataset
{
private:
	size_t batch_size;
	int num_batches;
	int size;
	float **inputs;
	float **labels;

	std::vector<Tensor> batches;
	std::vector<Tensor> targets;

public:
	SnakeDataset(int num_batches, size_t batch_size, string path_x, string path_y);

	int getNumOfBatches();
	int getSize();
	std::vector<Tensor> &getBatches();
	std::vector<Tensor> &getTargets();
};


int computeAccuracy(const Tensor &, const Tensor &, int);