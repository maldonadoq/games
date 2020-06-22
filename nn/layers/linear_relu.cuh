#pragma once

#include "layer.cuh"
#include "../utils/exception.cuh"
#include "../utils/shape.cuh"

class LinearReluLayer : public NNLayer {
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

	void initializeBiasWithZeros_Relu();
	void initializeWeightsRandomly_Relu();

  void ReluBackProp(Tensor& dZ);
	void computeAndStoreBackpropError_Relu(Tensor& dZ);
	void computeAndStoreLayerOutput_Relu(Tensor& A);
	void updateWeights_Relu(Tensor& dZ, float learning_rate);
	void updateBias_Relu(Tensor& dZ, float learning_rate);

public:
	LinearReluLayer(std::string name, Shape W_shape);
	~LinearReluLayer();

	Tensor& forward(Tensor& A);
	Tensor& backprop(Tensor& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Tensor getWeightsMatrix() const;
	Tensor getBiasVector() const;


};
