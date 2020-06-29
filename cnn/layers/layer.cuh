#pragma once

#include "../utils/tensor.cuh"
#include "../utils/utils.cuh"

#include <memory>
#include <vector>

class Layer {
 public:
	Layer() {}
	Layer(const Layer &other) = delete;
	Layer(Layer &&other) = delete;
	Layer &operator=(const Layer &other) = delete;
	Layer &operator=(Layer &&other) = delete;

	// connect to next layer
	Layer &connect(Layer &next_layer) {
		this->next = &next_layer;
		next_layer.pre = this;

		return next_layer;
	}

	virtual void forward() { throw std::runtime_error("not implement error"); };
	virtual void backward() { throw std::runtime_error("not implement error"); };

	// return pointer of weights and grads
	virtual std::vector<std::pair<Tensor *, Tensor *>> parameters() {
		throw std::runtime_error("not implement error");
	};

	virtual Tensor *get_grad() { return this->grad.get(); }
	virtual Tensor *get_output() { return this->output.get(); }

 protected:
	Layer *pre;
	Layer *next;

	// inputs grad and layer output
	std::unique_ptr<Tensor> grad;
	std::unique_ptr<Tensor> output;
};