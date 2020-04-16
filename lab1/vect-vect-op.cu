#include <iostream>
#include "src/vect-vect-op.cuh"
#include "src/utils.h"

using std::cout;
using std::endl;


int main(int argc, char const *argv[]){
	// Constant
	int threads = 512;

	// Size
	unsigned size_vector = 100;	

    float* a_host = new float[size_vector];
    float* b_host = new float[size_vector];
	float* r_host = new float[size_vector];

	for(int i=0; i<size_vector; i++){
        a_host[i] = i;
        b_host[i] = i;
	}
	
	operation(a_host, b_host, r_host, size_vector, threads, '-');	
	print_vector(r_host, size_vector);

	delete[] a_host;
    delete[] b_host;
    delete[] r_host;

	return 0;
}