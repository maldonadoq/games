#include "snake_dataset.cuh"

SnakeDataset::SnakeDataset(int num_batches, size_t batch_size, string path_x, string path_y){
	this->batch_size = batch_size;
	this->num_batches = num_batches;

	std::ifstream file;
	std::string line;

	unsigned i, j;
	int input_size, output_size;
	float tmp;

	file.open(path_x);
	this->size = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');

	file.seekg(0);
	std::getline(file, line);
	input_size = 1;
	for (i = 0; i < line.size(); i++){
		if (line[i] == ' '){
			input_size += 1;
		}
	}

	this->images = new float *[this->size];
	for (i = 0; i < this->size; i++){
		this->images[i] = new float[input_size];
	}

	i = 0;
	j = 0;
	while (!file.eof()){
		file >> tmp;
		this->images[i][j] = tmp;
		j += 1;
		i += (j == input_size) ? 1 : 0;
		j %= input_size;
	}

	file.close();

	// Labels

	file.open(path_y);
	file.seekg(0);
	std::getline(file, line);
	output_size = 1;
	for (i = 0; i < line.size(); i++){
		if (line[i] == ' '){
			output_size += 1;
		}
	}

	this->labels = new float *[this->size];
	for (int i = 0; i < this->size; i++){
		this->labels[i] = new float[output_size];
	}

	i = 0;
	j = 0;
	while (!file.eof()){
		file >> tmp;
		this->labels[i][j] = tmp;
		j += 1;
		i += (j == output_size) ? 1 : 0;
		j %= output_size;
	}

	file.close();

	for (int i = 0; i < num_batches; i++){
		batches.push_back(Matrix(Shape(batch_size, input_size)));
		targets.push_back(Matrix(Shape(batch_size, output_size)));

		batches[i].allocateMemory();
		targets[i].allocateMemory();

		for (int j = 0; j < batch_size; j++){
			int index = i * batch_size + j;
			for (int k = 0; k < input_size; k++){
				batches[i][k * batch_size + j] = this->images[index][k];
			}

			for (int k = 0; k < output_size; k++){
				targets[i][k * batch_size + j] = this->labels[index][k];
			}
		}

		batches[i].copyHostToDevice();
		targets[i].copyHostToDevice();
	}

	std::cout << batch_size * num_batches << std::endl;
}

int SnakeDataset::getNumOfBatches(){
	return num_batches;
}

std::vector<Matrix> &SnakeDataset::getBatches(){
	return batches;
}

std::vector<Matrix> &SnakeDataset::getTargets(){
	return targets;
}
