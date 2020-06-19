NVCC = nvcc
LIBS = -std=c++11 -lGL -lGLU -lglut

all: init

init:
	$(NVCC) -c ../nn/layers/linear_layer.cu ../nn/layers/linear_relu.cu ../nn/layers/linear_softmax.cu ../nn/layers/relu_activation.cu ../nn/layers/softmax_activation.cu
	$(NVCC) -c ../nn/neurals/neural_network.cu ../nn/neurals/snake_dataset.cu
	$(NVCC) -c ../nn/utils/cce_loss.cu ../nn/utils/matrix.cu ../nn/utils/shape.cu
	$(NVCC) -c src/result.cu src/snake.cpp src/utils.cpp
	$(NVCC) main.cu *.o -o main.out $(LIBS)

update:
	$(NVCC) -c src/snake.cpp src/utils.cpp
	$(NVCC) main.cu *.o -o main.out $(LIBS)

clean:
	rm *.o *.out
