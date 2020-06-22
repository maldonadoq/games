#include "result.cuh"

void printMatrix(const Tensor& m){
	for(int i = 0 ; i < m.shape.x ; i++){
		for(int j = 0 ; j < m.shape.y ; j++)
			std::cout << m[j * m.shape.x + i] << " ";
		std::cout << std::endl;
	}
}

std::vector<float> firstResultFloat(const Tensor &predictions, int k){
	int m = predictions.shape.x;
	std::vector<float> res;

    for(int i=0; i<k; i++){
		res.push_back(predictions[m*i]);
	}
	
	return res;
}

std::vector<int> firstResultInt(const Tensor &predictions, int k){
	int m = predictions.shape.x;
	std::vector<int> res(k, 0);

	int idx = 0;
	float tmp = predictions[idx];

    for(int i=1; i<k; i++){		
		if(predictions[m*i] > tmp){
			tmp = predictions[m*i];
			idx = i;
		}
	}

	res[idx] = 1;
	
	return res;
}

template<class T>
void printVector(const std::vector<T> &vec){
	for(unsigned i=0; i<vec.size(); i++){
		std::cout << vec[i] << " ";
	}
	std::cout << std::endl;
}

template void printVector<int>(const std::vector<int> &);
template void printVector<float>(const std::vector<float> &);