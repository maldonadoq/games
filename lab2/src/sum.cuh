#ifndef _SUM_H_
#define _SUM_H_

#define TILE 32

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
void sumColSharedKernel(int M, int N, int *Md, int *Rd){
	__shared__ int Mds[TILE][TILE];

	unsigned tx = threadIdx.x;
	unsigned ty = threadIdx.y;

	unsigned col = (blockIdx.x * blockDim.x) + tx;

	if(col < N){
		int tmp = 0;
		int row;

		for(int i=0; i<M/TILE; i++){
			row = (i * TILE) + ty;
			
			Mds[ty][tx] = Md[(row * N) + col];
			__syncthreads();

			for(int j=0; j<TILE; j++){
				tmp += Mds[ty][j];
			}
			__syncthreads();
		}

		Rd[col] = tmp;
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
	
	dim3 dimBlock(block, block);
	dim3 dimGrid(ceil((float)N / (float)block), ceil((float)M / (float)block));
	
	switch(t){
		case 'g': 
			sumColKernel<<<dimGrid, dimBlock>>>(M, N, Md, Rd);
			break;
		case 's':
			sumColSharedKernel<<<dimGrid, dimBlock>>>(M, N, Md, Rd);
			break;
		default:
			std::cout << "Type [s-g]!!!";
			break;
	}

	cudaMemcpy(Rh, Rd, size2, cudaMemcpyDeviceToHost);
	
	cudaFree(Md);
	cudaFree(Rd);
}

#endif