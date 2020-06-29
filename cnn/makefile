NVCC = nvcc
LIBS = -std=c++11

all: run

run:
	$(NVCC) -c layers/conv.cu layers/flatten.cu layers/linear.cu layers/max_pool.cu layers/nll_loss.cu layers/relu.cu layers/rmsprop.cu layers/softmax.cu
	$(NVCC) -c neurals/mnist.cu neurals/dataset.cu
	$(NVCC) -c utils/blas.cu utils/tensor.cu
	mv *.o bin
	$(NVCC) main.cu -o main.out bin/*.o

upd:
	$(NVCC) main.cu -o main.out bin/*.o

clean:
	rm bin/*.o *.out
