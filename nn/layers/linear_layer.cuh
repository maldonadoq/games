#pragma once
#include "layer.cuh"
#include "../utils/exception.cuh"
#include "../utils/shape.cuh"

class LinearLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	Tensor W;
	Tensor b;

	Tensor Z;
	Tensor A;
	Tensor dA;

	Tensor AT;
	Tensor WT;

	void initializeBiasWithZeros();
	void initializeWeightsRandomly();

	void computeAndStoreBackpropError(Tensor& dZ);
	void computeAndStoreLayerOutput(Tensor& A);
	void updateWeights(Tensor& dZ, float learning_rate);
	void updateBias(Tensor& dZ, float learning_rate);

public:
	LinearLayer(std::string name, Shape W_shape);
	~LinearLayer();

	Tensor& forward(Tensor& A);
	Tensor& backprop(Tensor& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Tensor getWeightsMatrix() const;
	Tensor getBiasVector() const;
};
