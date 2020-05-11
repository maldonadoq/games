#ifndef _SUM_H_
#define _SUM_H_

#define TILE 32

#include <iostream>

using std::cout;
using std::endl;

__global__
void sumColKernel(int M, int N, int* Md, int* Rd){

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(row < M and col < N){
		int tmp = 0;

		for (int i=0; i<M; i++) {
			tmp += Md[(N * i) + col];
		}
	
		Rd[col] = tmp;
	}
}

__global__
void sumColSharedKernel(int M, int N, int* Md, int* Nd){
	__shared__ int Nds[TILE];

	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	Nds[threadIdx.y] = Md[(row*N) + col];
	__syncthreads();

	if (threadIdx.y == 0){
		for (int i = 1; i < blockDim.y; ++i) {
			Nds[0] = Nds[0]+Nds[i];
		}
		
		Nd[blockIdx.x] += Nds[0];
	}
}

void sumCol(int M, int N, int *Mh, int *Rh, int block, char t){
	int size1 = M * N * sizeof(int);
	int size2 = N * sizeof(int);

	int* Md;
	int* Rd;

	cudaMalloc(&Md, size1);
	cudaMalloc(&Rd, size2);

	cudaMemcpy(Md, Mh, size1, cudaMemcpyHostToDevice);
	cudaMemset(Rd, 0, size2);
		
	switch(t){
		case 'g': {
			dim3 dimBlock(block, block);
			dim3 dimGrid(ceil((float)N / (float)block), ceil((float)M / (float)block));
			sumColKernel<<<dimGrid, dimBlock>>>(M, N, Md, Rd);
			break;
		}
		case 's':{
			dim3 dimGrid(N, M/block);
			dim3 dimBlock(1, block);
			sumColSharedKernel<<<dimGrid, dimBlock>>>(M, N, Md, Rd);
			break;
		}
		default:
			std::cout << "Type [s-g]!!!";
			break;
	}

	cudaMemcpy(Rh, Rd, size2, cudaMemcpyDeviceToHost);
	
	cudaFree(Md);
	cudaFree(Rd);
}

#endif