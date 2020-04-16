#include <iostream>

using std::cout;
using std::endl;

void printDeviceInfo(cudaDeviceProp prop, int idx){
	cout << "[" << idx << "]\n";
	cout << "  Name: " << prop.name << endl;
	cout << "  Major: " << prop.major << endl;
	cout << "  Minor: " << prop.major << endl;
	cout << "  Total Global Memory: " << prop.totalGlobalMem << endl;
	cout << "  Total Shared Memory per Block: " << prop.sharedMemPerBlock << endl;

	int i;
	int dim = 3;

	cout << "  Maximun Block Dim: \n";
	for(i=0; i<dim; i++){
		cout << "    Dim " << i << ": " << prop.maxThreadsDim[i] << endl;
	}

	cout << "  Maximun Grid Dim: \n";
	for(i=0; i<dim; i++){
		cout << "    Dim " << i << ": " << prop.maxGridSize[i] << endl;
	}

	cout << "  Warp Size: " << prop.warpSize << endl;
	cout << "  Maximun Threads per Block: " << prop.maxThreadsPerBlock << endl;
	cout << "  Number of Multiprocessors: " << prop.multiProcessorCount << endl;	
}

int main(int argc, char const *argv[]){

	cudaDeviceProp prop;

	int count = 0;
	int i;

	cudaGetDeviceCount(&count);

	for(i=0; i<count ;i++ ){
		cudaGetDeviceProperties(&prop, i);
		printDeviceInfo(prop, i);
		cout << endl;
	}

	return 0;
}