#pragma once
#include "../utils/blas.cuh"
#include "optimizer.cuh"

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <memory>
#include <unordered_map>
#include <vector>

class RMSProp : public Optimizer {
 public:
	explicit RMSProp(float learning_rate = 0.01, float l2 = 0.001,
									 float beta = 0.99)
			: learning_rate(learning_rate), l2(l2), beta(beta) {
		std::cout << "learning rate: " << learning_rate << ", l2: " << l2
							<< ", beta: " << beta << std::endl;
	}

	void regist(std::vector<std::pair<Tensor *, Tensor *>> params);
	void step();

 private:
	std::vector<std::unique_ptr<Tensor>> square_grad;

	float learning_rate;
	float l2;
	float beta;
};