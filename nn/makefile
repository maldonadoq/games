NVCC = nvcc
LIBS = -std=c++11

all: run

run:
	$(NVCC) -c main.cu
	$(NVCC) -c layers/linear_layer.cu layers/linear_relu.cu layers/linear_softmax.cu layers/relu_activation.cu layers/softmax_activation.cu
	$(NVCC) -c neurals/neural_network.cu neurals/snake_dataset.cu
	$(NVCC) -c utils/cce_loss.cu utils/matrix.cu utils/shape.cu
	$(NVCC) -o main.out *.o

clean:
	rm *.o *.out