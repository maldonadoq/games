# Neural Network
Neural Network [Perceptron Multilayer in Cuda]

## Code

### Main

```c++
NeuralNetwork nn;

nn.addLayer(new LinearLayer("Linear_1", Shape(input_size, 256)));
nn.addLayer(new ReLUActivation("Relu_1"));
nn.addLayer(new LinearLayer("Linear_2", Shape(256, 128)));
nn.addLayer(new ReLUActivation("Relu_2"));
nn.addLayer(new LinearLayer("Linear_3", Shape(128, output_size)));
nn.addLayer(new softmaxActivation("Softmax"));

nn.summary();
```

### Training

```c++
SnakeDataset snake(num_batches_train, batch_size, "data/trainX.csv", "data/trainY.csv");

for (epoch = 0; epoch < epochs; epoch++){
    loss = 0.0;
    for (batch = 0; batch < num_batches_train; batch++){
        Y = nn.forward(snake.getBatches().at(batch));
        nn.backprop(Y, snake.getTargets().at(batch));

        correct += computeAccuracy(Y, snake.getTargets().at(batch), output_size);
        loss += cce_cost.cost(Y, snake.getTargets().at(batch));
    }
}
```

### Testing

```c++
SnakeDataset snake_test(num_batches_test, batch_size, "data/testX.csv", "data/testY.csv");
correct = 0;

for (batch = 0; batch < num_batches_test; batch++){
    Y = nn.forward(snake_test.getBatches().at(batch));
    Y.copyDeviceToHost();
    correct += computeAccuracy(Y, snake_test.getTargets().at(batch), output_size);
}
```

### Neural Network

```c++
class NeuralNetwork {
private:
	vector<NNLayer*> layers;
	CCELoss cce_cost;

	Tensor Y;
	Tensor dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Tensor forward(Tensor);
	void backprop(Tensor, Tensor);

	void addLayer(NNLayer *);
	std::vector<NNLayer*> getLayers() const;

	bool save(string);
	bool load(string);

	void summary();
};
```

### Dataset

```c++
class SnakeDataset{
private:
    size_t batch_size;
    int num_batches;
    int size;
    float **inputs;
    float **labels;

    std::vector<Tensor> batches;
    std::vector<Tensor> targets;

public:
    SnakeDataset(int, size_t, string, string);

    int getNumOfBatches();
    int getSize();
    std::vector<Tensor> &getBatches();
    std::vector<Tensor> &getTargets();
};
```

## How to run

```bash
$ cd folder name
$ make
$ ./main.out epochs # 71
```

## Images
*   **100 Epochs - Loss Evolution** :
<p align="center"> 
<img src="https://github.com/maldonadoq/games/blob/master/nn/img/loss.png" width="500">
</p>