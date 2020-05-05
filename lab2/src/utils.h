#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>

using std::cout;
using std::endl;

template<class T>
void print_matrix(T *m, unsigned width, unsigned height){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			cout << m[(i*width) + j] << " ";
		}
		cout << endl;
	}
}

template<class T>
void print_vector(T *v, unsigned s){
	for(unsigned i=0; i<s; i++){
        std::cout << v[i] << " ";
    }
	std::cout << "\n";
}

#endif