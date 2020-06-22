#include "tensor.cuh"

Tensor::Tensor(size_t x_dim, size_t y_dim) : shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
											 device_allocated(false), host_allocated(false){
}

Tensor::Tensor(Shape shape) : Tensor(shape.x, shape.y){
}

Tensor::Tensor(Shape shape, std::vector<float> input) : Tensor(shape.x, shape.y){
	allocateMemory();
	for(int i=0; i<input.size(); i++){
		data_host.get()[i] = input[i];
	}
	copyHostToDevice();
}

void Tensor::allocateCudaMemory(){
	if (!device_allocated){
		float *device_memory = nullptr;
		cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
		NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
		data_device = std::shared_ptr<float>(device_memory,
											 [&](float *ptr) { cudaFree(ptr); });
		device_allocated = true;
	}
}

void Tensor::allocateHostMemory(){
	if (!host_allocated){
		data_host = std::shared_ptr<float>(new float[shape.x * shape.y],
										   [&](float *ptr) { delete[] ptr; });
		host_allocated = true;
	}
}

void Tensor::allocateMemory(){
	allocateCudaMemory();
	allocateHostMemory();
}

void Tensor::allocateMemoryIfNotAllocated(Shape shape){
	if (!device_allocated && !host_allocated){
		this->shape = shape;
		allocateMemory();
	}
}

void Tensor::copyHostToDevice(){
	if (device_allocated && host_allocated){
		cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	}
	else{
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

void Tensor::copyDeviceToHost(){
	if (device_allocated && host_allocated){
		cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
	}
	else{
		throw NNException("Cannot copy device data to not allocated memory on host.");
	}
}

float &Tensor::operator[](const int index){
	return data_host.get()[index];
}

const float &Tensor::operator[](const int index) const{
	return data_host.get()[index];
}
