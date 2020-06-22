#include "softmax_activation.cuh"

softmaxActivation::softmaxActivation(std::string name){
	this->name = name;
}

softmaxActivation::~softmaxActivation(){
}

__global__ void softmax_trivial(float *softmaxP, float *b, int rows, int cols){
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	float _max = -100000000.0;
	float sum = 0.0;

	if (tid * cols + bid < rows * cols){
		for (int i = 0; i < rows; i++)
			_max = max(_max, b[i * cols + bid]);
		for (int i = 0; i < rows; i++)
			softmaxP[i * cols + bid] = (b[i * cols + bid] - _max);
		for (int i = 0; i < rows; i++)
			softmaxP[i * cols + bid] = __expf(softmaxP[i * cols + bid]);
		for (int i = 0; i < rows; i++)
			sum += softmaxP[i * cols + bid];
		for (int i = 0; i < rows; i++)
			softmaxP[i * cols + bid] /= sum;
	}
}

/*
  * blocks : cuSoftMaxP->rows
  * threads: cuSoftMaxP->cols
  * shared : sizeof(float) * cuSoftMaxP->cols * 2
  */
__global__ void g_getSoftMaxP(float *softMaxP, float *b, int cols, int row){
	int bid = blockIdx.x;
	extern __shared__ float _share[];
	float *_max = _share;
	float *_sum = _share + blockDim.x;
	float *sp = softMaxP + bid;
	_sum[threadIdx.x] = 0.0;
	_max[threadIdx.x] = -100000000.0;
	for (int tid = threadIdx.x * cols + blockIdx.x; tid < row * cols; tid += cols){
		sp[tid] += b[tid];
		_max[threadIdx.x] = max(_max[threadIdx.x], sp[tid]);
	}
	__syncthreads();
	int len = blockDim.x;
	while (len != 1){
		__syncthreads();
		int skip = (len + 1) >> 1;
		if (threadIdx.x < (len >> 1))
		{
			if (_max[threadIdx.x] < _max[threadIdx.x + skip])
			{
				_max[threadIdx.x] = _max[threadIdx.x + skip];
			}
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();
	for (int tid = threadIdx.x * cols + blockIdx.x; tid < row * cols; tid += cols){
		sp[tid] -= _max[0];
		sp[tid] = __expf(sp[tid]);
		_sum[threadIdx.x] += sp[tid];
	}
	__syncthreads();
	len = blockDim.x;
	while (len != 1){
		__syncthreads();
		int skip = (len + 1) >> 1;
		if (threadIdx.x < (len >> 1))
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();
	for (int tid = threadIdx.x * cols + blockIdx.x; tid < row * cols; tid += cols){
		sp[tid] /= _sum[0];
	}
}

Tensor &softmaxActivation::forward(Tensor &Z){
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);
	dim3 block = A.shape.x;
	dim3 threads = 1;
	softmax_trivial<<<block, threads>>>(A.data_device.get(), Z.data_device.get(), A.shape.y, A.shape.x);

	cudaStreamSynchronize(0);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax forward propagation.");

	return A;
}

__global__ void softmaxActivationBackprop(float *Z, float *dA, float *dZ,
										  int Z_x_dim, int Z_y_dim){

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim){
		dZ[index] = dA[index];
	}
}

Tensor &softmaxActivation::backprop(Tensor &dA, float learning_rate){
	dZ.allocateMemoryIfNotAllocated(Z.shape);
	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	softmaxActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device.get(), dA.data_device.get(),
															 dZ.data_device.get(),
															 Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform softmax back propagation");

	return dZ;
}
