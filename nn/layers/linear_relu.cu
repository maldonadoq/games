#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "linear_relu.cuh"

#define BLOCK_DIM 16

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory
// so that bank conflicts do not occur when threads address the array column-wise.
__global__ void transpose_relu(float *odata, float *idata, int width, int height) {
	__shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

	// read the matrix tile into shared memory
	// load one element per thread from device memory (idata) and store it
	// in transpose_relud order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if ((xIndex < width) && (yIndex < height)) {
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	// synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transpose_relud matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if ((xIndex < height) && (yIndex < width)) {
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyAndRelu(float *A, float *B, float *C, float *b, float *T,
									  int numARows, int numAColumns,
									  int numBRows, int numBColumns,
									  int numCRows, int numCColumns) {
	// Insert code to implement matrix multiplication here
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		Row = by * TILE_WIDTH + ty,
		Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {
		if (Row < numARows && m * TILE_WIDTH + tx < numAColumns)
			ds_M[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
		else
			ds_M[ty][tx] = 0;
		if (Col < numBColumns && m * TILE_WIDTH + ty < numBRows)
			ds_N[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
		else
			ds_N[ty][tx] = 0;

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += ds_M[ty][k] * ds_N[k][tx];
		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns) {
		float num = (Pvalue + b[Row]);
		T[Row * numCColumns + Col] = num;
		C[Row * numCColumns + Col] = fmaxf(num, 0);
	}
}

__global__ void matrixMultiplyBackPropRelu(float *A, float *B, float *C,
										   int numARows, int numAColumns,
										   int numBRows, int numBColumns,
										   int numCRows, int numCColumns) {
	// Insert code to implement matrix multiplication here
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		Row = by * TILE_WIDTH + ty,
		Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {
		if (Row < numARows && m * TILE_WIDTH + tx < numAColumns)
			ds_M[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
		else
			ds_M[ty][tx] = 0;
		if (Col < numBColumns && m * TILE_WIDTH + ty < numBRows)
			ds_N[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
		else
			ds_N[ty][tx] = 0;

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += ds_M[ty][k] * ds_N[k][tx];
		__syncthreads();
	}

	if (Row < numCRows && Col < numCColumns)
		C[Row * numCColumns + Col] = Pvalue;
}

// Compute C = A * B
__global__ void matrixMultiplyUpdateWeights_relu(float *A, float *B, float *C,
												 int numARows, int numAColumns,
												 int numBRows, int numBColumns,
												 int numCRows, int numCColumns,
												 float learning_rate) {
	// Insert code to implement matrix multiplication here
	__shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		Row = by * TILE_WIDTH + ty,
		Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {
		if (Row < numARows && m * TILE_WIDTH + tx < numAColumns)
			ds_M[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
		else
			ds_M[ty][tx] = 0;
		if (Col < numBColumns && m * TILE_WIDTH + ty < numBRows)
			ds_N[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
		else
			ds_N[ty][tx] = 0;

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += ds_M[ty][k] * ds_N[k][tx];
		__syncthreads();
	}
	if (Row < numCRows && Col < numCColumns)
		C[Row * numCColumns + Col] = C[Row * numCColumns + Col] - learning_rate * (Pvalue / numAColumns);
}

__global__ void initializeBiasKernel_relu(float *b, int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		b[index] = 0.0;
	}
}

__global__ void updateBiasKernel_relu(float *dZ, float *b, int cols, int row, float learning_rate) {
	int bid = blockIdx.x;
	extern __shared__ float _share[];
	//float * _max = _share;
	float *_sum = _share;
	float *sp = dZ + cols * bid;
	_sum[threadIdx.x] = 0.0;

	for (int id = threadIdx.x; id < cols; id += blockDim.x) {
		//	int id = tid + threadIdx.x;
		//if(id < cols){
		_sum[threadIdx.x] += sp[id];
		//}
	}
	__syncthreads();
	int len = blockDim.x;
	while (len != 1) {
		__syncthreads();
		int skip = (len + 1) >> 1;
		if (threadIdx.x < (len >> 1))
		{
			_sum[threadIdx.x] += _sum[threadIdx.x + skip];
		}
		len = (len + 1) >> 1;
	}
	__syncthreads();
	b[bid] -= learning_rate * (_sum[0] / cols);
}

LinearReluLayer::LinearReluLayer(std::string name, Shape W_shape) : W(W_shape), b(W_shape.y, 1) {
	this->name = name;
	b.allocateMemory();
	W.allocateMemory();
	initializeBiasWithZeros_Relu();
	initializeWeightsRandomly_Relu();
}

LinearReluLayer::~LinearReluLayer() {
}

void LinearReluLayer::initializeWeightsRandomly_Relu() {

	std::default_random_engine generator;
	std::normal_distribution<float> normal_distribution(0.0, 1.0);
	float weights_init_threshold = 0.01;

	for (int x = 0; x < W.shape.x; x++) {
		for (int y = 0; y < W.shape.y; y++)
		{
			W[y * W.shape.x + x] = normal_distribution(generator) * weights_init_threshold;
		}
	}

	W.copyHostToDevice();
}

void LinearReluLayer::initializeBiasWithZeros_Relu() {
	dim3 blockDim(256);
	dim3 gridDim((b.shape.x * b.shape.y + blockDim.x - 1) / blockDim.x);

	initializeBiasKernel_relu<<<gridDim, blockDim>>>(b.data_device.get(), b.shape.x * b.shape.y);
}

Tensor &LinearReluLayer::forward(Tensor &A) {
	assert(W.shape.x == A.shape.y);

	this->A = A;
	Shape Z_shape(A.shape.x, W.shape.y);
	Z.allocateMemoryIfNotAllocated(Z_shape);

	T.allocateMemoryIfNotAllocated(Z_shape);

	computeAndStoreLayerOutput_Relu(A);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform linear layer forward propagation.");

	return Z;
}

void LinearReluLayer::computeAndStoreLayerOutput_Relu(Tensor &A) {
	dim3 block_size(TILE_WIDTH, TILE_WIDTH);
	dim3 num_of_blocks((Z.shape.x + block_size.x - 1) / block_size.x,
					   (Z.shape.y + block_size.y - 1) / block_size.y);

	matrixMultiplyAndRelu<<<num_of_blocks, block_size>>>(W.data_device.get(),
														 A.data_device.get(),
														 Z.data_device.get(),
														 b.data_device.get(),
														 T.data_device.get(),
														 W.shape.y, W.shape.x,
														 A.shape.y, A.shape.x,
														 Z.shape.y, Z.shape.x);
}

__global__ void ReluBackKernel(float *Z, float *dZ, int size) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (Z[id] <= 0)
			dZ[id] = 0;
	}
}

Tensor &LinearReluLayer::backprop(Tensor &dZ, float learning_rate) {
	dA.allocateMemoryIfNotAllocated(A.shape);
	WT.allocateMemoryIfNotAllocated(Shape(W.shape.y, W.shape.x));
	AT.allocateMemoryIfNotAllocated(Shape(A.shape.y, A.shape.x));

	ReluBackProp(dZ);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform back propagation Relu Fusion.");

	computeAndStoreBackpropError_Relu(dZ);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform back propagation.");

	updateBias_Relu(dZ, learning_rate);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform bias update.");

	updateWeights_Relu(dZ, learning_rate);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform weights update.");

	return dA;
}

void LinearReluLayer::ReluBackProp(Tensor &dZ) {
	dim3 block_size(256);
	dim3 num_of_block((Z.shape.x * Z.shape.y + block_size.x - 1) / block_size.x);
	ReluBackKernel<<<num_of_block, block_size>>>(Z.data_device.get(), dZ.data_device.get(), dZ.shape.x * dZ.shape.y);
}

void LinearReluLayer::computeAndStoreBackpropError_Relu(Tensor &dZ) {
	dim3 block_size(TILE_WIDTH, TILE_WIDTH);
	dim3 num_of_blocks((A.shape.x + block_size.x - 1) / block_size.x,
					   (A.shape.y + block_size.y - 1) / block_size.y);

	dim3 transpose_relu_block(BLOCK_DIM, BLOCK_DIM);
	dim3 num_t_blocks((W.shape.x + transpose_relu_block.x - 1) / transpose_relu_block.x,
					  (W.shape.y + transpose_relu_block.y - 1) / transpose_relu_block.y);
	transpose_relu<<<num_t_blocks, transpose_relu_block>>>(WT.data_device.get(), W.data_device.get(), W.shape.x, W.shape.y);

	matrixMultiplyBackPropRelu<<<num_of_blocks, block_size>>>(WT.data_device.get(),
															  dZ.data_device.get(),
															  dA.data_device.get(),
															  WT.shape.y, WT.shape.x,
															  dZ.shape.y, dZ.shape.x,
															  dA.shape.y, dA.shape.x);
}

void LinearReluLayer::updateWeights_Relu(Tensor &dZ, float learning_rate) {
	dim3 block_size(TILE_WIDTH, TILE_WIDTH);
	dim3 num_of_blocks((W.shape.x + block_size.x - 1) / block_size.x,
					   (W.shape.y + block_size.y - 1) / block_size.y);

	dim3 transpose_relu_block(BLOCK_DIM, BLOCK_DIM);
	dim3 num_t_blocks((A.shape.x + transpose_relu_block.x - 1) / transpose_relu_block.x,
					  (A.shape.y + transpose_relu_block.y - 1) / transpose_relu_block.y);
	transpose_relu<<<num_t_blocks, transpose_relu_block>>>(AT.data_device.get(), A.data_device.get(), A.shape.x, A.shape.y);

	matrixMultiplyUpdateWeights_relu<<<num_of_blocks, block_size>>>(dZ.data_device.get(),
																	AT.data_device.get(),
																	W.data_device.get(),
																	dZ.shape.y, dZ.shape.x,
																	AT.shape.y, AT.shape.x,
																	W.shape.y, W.shape.x,
																	learning_rate);
}

void LinearReluLayer::updateBias_Relu(Tensor &dZ, float learning_rate) {
	dim3 block_size(std::min(256, int(dZ.shape.x)));
	dim3 num_of_blocks(dZ.shape.y);
	updateBiasKernel_relu<<<num_of_blocks, block_size, sizeof(float) * block_size.x>>>(dZ.data_device.get(), b.data_device.get(), dZ.shape.x, dZ.shape.y, learning_rate);
}

int LinearReluLayer::getXDim() const {
	return W.shape.x;
}

int LinearReluLayer::getYDim() const {
	return W.shape.y;
}

Tensor LinearReluLayer::getWeightsMatrix() const {
	return W;
}

Tensor LinearReluLayer::getBiasVector() const {
	return b;
}
