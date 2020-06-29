#include "neurals/dataset.cuh"
#include "neurals/mnist.cuh"

#define batch_size 512

int main(int argc, char *argv[]){
	int epochs = 3;

	Mnist mnist("data", 0.003, 0.0001, 0.99);
	
	mnist.train(epochs, batch_size);
	mnist.test(batch_size);
}
