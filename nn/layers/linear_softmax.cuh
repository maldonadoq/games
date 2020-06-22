#pragma once
#include "layer.cuh"
#include "../utils/exception.cuh"
#include "../utils/shape.cuh"


class LinearSoftmaxLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	Tensor W;
	Tensor b;

	Tensor T;
	Tensor Z;
	Tensor A;
	Tensor dA;

	Tensor AT;
	Tensor WT;

	void initializeBiasWithZeros_softmax();
	void initializeWeightsRandomly_softmax();

	void computeAndStoreBackpropError_softmax(Tensor& dZ);
	void computeAndStoreLayerOutput_softmax(Tensor& A);
	void updateWeights_softmax(Tensor& dZ, float learning_rate);
	void updateBias_softmax(Tensor& dZ, float learning_rate);


public:
	LinearSoftmaxLayer(std::string name, Shape W_shape);
	~LinearSoftmaxLayer();

	Tensor& forward(Tensor& A);
	Tensor& backprop(Tensor& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Tensor getWeightsMatrix() const;
	Tensor getBiasVector() const;


};
