#include "result.cuh"

void printMatrix(const Matrix& m){
	for(int i = 0 ; i < m.shape.x ; i++){
		for(int j = 0 ; j < m.shape.y ; j++)
			std::cout << m[j * m.shape.x + i] << " ";
		std::cout << std::endl;
	}
}

int computeAccuracy(const Matrix &predictions, const Matrix &targets, int k){
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++){
		float _max = 0.0;
		float _maxt = 0.0;
		int label = 0;
		int labely = 0;
		for (int j = 0; j < k; j++){
			if (predictions[j * m + i] > _max){
				_max = predictions[j * m + i];
				label = j;
			}
			if (targets[j * m + i] > _maxt){
				_maxt = targets[j * m + i];
				labely = j;
			}
		}
		if (label == labely)
			correct_predictions++;
	}

	return correct_predictions;
}

std::vector<float> firstResult(const Matrix &predictions, int k){
	int m = predictions.shape.x;
	std::vector<float> res;

    for(int i=0; i<k; i++){
		res.push_back(predictions[m*i]);
	}
	
	return res;
}

void printVector(const std::vector<float> &vec){
	for(unsigned i=0; i<vec.size(); i++){
		std::cout << vec[i] << " ";
	}
	std::cout << std::endl;
}