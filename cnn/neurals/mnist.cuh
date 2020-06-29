#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "dataset.cuh"

#include "../utils/blas.cuh"
#include "../utils/tensor.cuh"

#include "../layers/conv.cuh"
#include "../layers/flatten.cuh"
#include "../layers/linear.cuh"
#include "../layers/max_pool.cuh"
#include "../layers/nll_loss.cuh"
#include "../layers/relu.cuh"
#include "../layers/rmsprop.cuh"
#include "../layers/softmax.cuh"

class Mnist {
 public:
	explicit Mnist(std::string mnist_data_path, float learning_rate, float l2,
					float beta);

	void train(int epochs, int batch_size);
	void test(int batch_size);

 private:
	void forward(int batch_size, bool is_train);  // neural network forward
	void backward();                              // neural network backward

	std::pair<int, int> get_accuracy(
		const thrust::host_vector<
			float, thrust::system::cuda::experimental::pinned_allocator<float>>&
			probs,
		int cls_size,
		const thrust::host_vector<
			float, thrust::system::cuda::experimental::pinned_allocator<float>>&
			labels);  // get_accuracy

	// Conv1_5x5     1 * 32
	// MaxPool1_2x2
	// Conv2_5x5     32 * 64
	// MaxPool2_2x2
	// Conv3_3x3     64 * 128
	// FC1           (128 *  2 * 2) * 128
	// SoftMax
	// NLLLoss

	std::unique_ptr<RMSProp> rmsprop;
	std::unique_ptr<DataSet> dataset;

	std::unique_ptr<Conv> conv1;
	std::unique_ptr<ReLU> conv1_relu;
	std::unique_ptr<MaxPool> max_pool1;

	std::unique_ptr<Conv> conv2;
	std::unique_ptr<ReLU> conv2_relu;
	std::unique_ptr<MaxPool> max_pool2;

	std::unique_ptr<Conv> conv3;
	std::unique_ptr<ReLU> conv3_relu;
	std::unique_ptr<Flatten> flatten;

	std::unique_ptr<Linear> fc1;
	std::unique_ptr<ReLU> fc1_relu;

	std::unique_ptr<Softmax> softmax;
	std::unique_ptr<NLLLoss> nll_loss;
};
