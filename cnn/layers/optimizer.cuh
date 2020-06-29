#pragma once

#include <utility>
#include <vector>

#include "../utils/tensor.cuh"

class Optimizer {
 public:
	virtual void step() = 0;
	virtual void regist(std::vector<std::pair<Tensor *, Tensor *>> params) = 0;

 protected:
	std::vector<Tensor *> parameter_list;
	std::vector<Tensor *> grad_list;
};