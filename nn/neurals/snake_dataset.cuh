#pragma once

#include "../utils/matrix.cuh"
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
	float **images;
	float **labels;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:
	SnakeDataset(int num_batches, size_t batch_size, string path_x, string path_y);

	int getNumOfBatches();
	int getSize();
	std::vector<Matrix> &getBatches();
	std::vector<Matrix> &getTargets();
};
