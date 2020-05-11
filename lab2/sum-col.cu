#include <iostream>
#include "src/sum.cuh"
#include "src/utils.h"

using std::cout;
using std::endl;


int main(int argc, char const *argv[]){
	int M = 512;
	int N = 1024;
	int block = 32;

    int* Mh = new int[M * N];
	int* Rh = new int[N];

	for(int i=0; i<M; i++){
		for(int j=0; j<N; j++){
			Mh[(i * N) + j] = 1;
		}
	}
	
	// print_matrix(Mh, N, M);
	// cout << endl;

	sumCol(M, N, Mh, Rh, block, 's');
	print_vector(Rh, N);

	delete[] Mh;
    delete[] Rh;

	return 0;
}