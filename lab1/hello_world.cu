#include <iostream>

__global__ void helloWorld(){
	
}

int main(int argc, char const *argv[]){
	
	helloWorld<<< 1,1 >>>();

	return 0;
}