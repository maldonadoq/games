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
void sumColSharedKernel(int* Md, int* Nd, int M){
	__shared__ int Nds[TILE];

	int tmp = 0;
	int steps = M/blockDim.x;
	int init = blockIdx.x * M + threadIdx.x * steps;
	
	for(int k=0; k<steps; k++) {
		tmp = tmp + Md[init + k];
	}

	Nds[threadIdx.x] = tmp;
	__syncthreads();

	if (threadIdx.x == 0){
		for (int i = 1; i < blockDim.x; ++i) {
			Nds[0] = Nds[0]+Nds[i];
		}
		Nd[blockIdx.x] = Nds[0];
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
			dim3 dimGrid(N, 1);
			dim3 dimBlock(M/block, 1);
			sumColSharedKernel<<<dimGrid, dimBlock>>>(Md, Rd, M);
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