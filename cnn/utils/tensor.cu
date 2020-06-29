#include "tensor.cuh"

Tensor::Tensor(const std::vector<int> &_shape) : shape(_shape) {
	int size = 1;
	for (int i = 0; i < _shape.size(); i++) {
		size *= _shape[i];
	}

	this->data.resize(size);
}

Tensor::Tensor(const std::vector<int> &_shape, float value) : shape(_shape) {
	int size = 1;
	for (int i = 0; i < _shape.size(); i++) {
		size *= _shape[i];
	}

	this->data.resize(size, value);
}

Tensor::Tensor(const std::vector<int> &_shape,
								 const std::vector<float> &_data)
		: shape(_shape), data(_data.begin(), _data.end()) {
	this->check_size();
}

Tensor::Tensor(const Tensor &other) { *this = other; }

Tensor &Tensor::operator=(const Tensor &other) {
	if (this != &other) {
		this->shape = other.shape;
		this->data = other.data;
	}

	return *this;
}

Tensor::Tensor(Tensor &&other) { *this = std::move(other); }

Tensor &Tensor::operator=(Tensor &&other) {
	if (this != &other) {
		this->shape = std::move(other.shape);
		this->data = std::move(other.data);
	}
	return *this;
}

void Tensor::reshape(const std::vector<int> &_shape) {
	this->shape = _shape;
	this->check_size();
}

void Tensor::resize(const std::vector<int> &_shape) {
	this->shape = _shape;

	int size = 1;
	for (int i = 0; i < _shape.size(); i++) {
		size *= _shape[i];
	}

	if (size != this->data.size()) {
		this->data.resize(size);
	}
}

__global__ void storage_xavier(float *a, int size, float scale,
															 curandState *cs) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		curand_init(1234, index, 0, &cs[index]);
		a[index] = (curand_uniform(&cs[index]) * 2 - 1) * scale;
	}
}

void Tensor::xavier(size_t in_size, size_t out_size) {
	float *a_ptr = RAW_PTR(this->data);
	int size = this->data.size();
	int grid_size = ceil((float)(size) / BLOCK_SIZE);

	thrust::device_vector<curandState> cs(size);
	curandState *cs_ptr = RAW_PTR(cs);
	float scale = std::sqrt((float)6) / std::sqrt((float)(in_size) + out_size);
	storage_xavier<<<grid_size, BLOCK_SIZE>>>(a_ptr, size, scale, cs_ptr);

	CUDA_POST_KERNEL_CHECK;
}

void Tensor::check_size() {
	int size = 1;
	for (int i = 0; i < this->shape.size(); i++) {
		size *= this->shape[i];
	}
	CHECK_EQ(size, this->data.size(), "Tensor: size error");
}