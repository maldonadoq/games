#ifndef _VECTOR_CONST_OP_H_
#define _VECTOR_CONST_OP_H_

// Kernel of Multiplication
__global__
void kernelMult(float* a_device, int constant, float *r_device, int size_vector){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < size_vector){
		r_device[index] = a_device[index] * constant;
	}
}

// Kernel of Addition
__global__
void kernelPlus(float* a_device, int constant, float *r_device, int size_vector){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < size_vector){
		r_device[index] = a_device[index] + constant;
	}
}

// Kernel of Substraction
__global__
void kernelMinus(float* a_device, int constant, float *r_device, int size_vector){
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < size_vector){
		r_device[index] = a_device[index] - constant;
	}
}



// Operation [vector *+- constante]
void operation(float *a_host, int constant, float *r_host, int size_vector, int threads, char op){
	int size_device = size_vector*sizeof(float);
	float *a_device, *r_device;

	int blocks = ceil(size_vector/float(threads));

	cudaMalloc((void **)&a_device, size_device);
	cudaMalloc((void **)&r_device, size_device);

	cudaMemcpy(a_device, a_host, size_device, cudaMemcpyHostToDevice);
	
	switch(op){
		case '*':
			kernelMult<<<blocks, threads>>>(a_device, constant, r_device, size_vector);
			break;
		case '+':
			kernelPlus<<<blocks, threads>>>(a_device, constant, r_device, size_vector);
			break;
		case '-':
			kernelMinus<<<blocks, threads>>>(a_device, constant, r_device, size_vector);
			break;
		default:
			break;
	}

	cudaDeviceSynchronize();

	cudaMemcpy(r_host, r_device, size_device, cudaMemcpyDeviceToHost);
	
	cudaFree(a_device);
	cudaFree(r_device);
}

#endif