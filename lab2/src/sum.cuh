#ifndef _SUM_H_
#define _SUM_H_

__global__
void sumColKernel(int M, int* Md, int* Rd){

	int tval = 0;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int posIni = col * M;
	
	for (int k=0; k<M; ++k) {
		tval = tval + Md[posIni + k];
	}

	Rd[col] = tval;
}

void sumCol(int M, int N, int *Mh, int *Rh, int block){
	int size1 = M * N * sizeof(int);
	int size2 = N * sizeof(int);

	int* Md;
	int* Rd;

	cudaMalloc(&Md, size1);
	cudaMalloc(&Rd, size2);

	cudaMemcpy(Md, Mh, size1, cudaMemcpyHostToDevice);
	cudaMemset(Rd, 0, size2);
	
	int blocks = ceil(N / block);

	dim3 dimGrid(blocks, 1);
	dim3 dimBlock(block, 1, 1);
	
	sumColKernel<<<dimGrid, dimBlock>>>(M, Md, Rd);

	cudaMemcpy(Rh, Rd, size2, cudaMemcpyDeviceToHost);
	
	cudaFree(Md);
	cudaFree(Rd);
}

#endif