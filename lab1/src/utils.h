#ifndef _UTILS_H_
#define _UTILS_H_

template<class T>
void print_vector(T *v, unsigned s){
	for(unsigned i=0; i<s; i++){
        std::cout << v[i] << " ";
    }
	std::cout << "\n";
}

#endif