#include "layer.h"
#include <iostream>

layer::layer(int inputs, int neurons_count) {
	weights = Eigen::MatrixXd::Random(inputs, neurons_count);
	biases = Eigen::VectorXd::Zero(neurons_count);
 
	std::cout << "layer created with : " << inputs << " , "<<neurons_count <<std::endl;
}

void layer::forward(const Eigen::VectorXd& input) {
	neurons_values = (weights.transpose() * input) + biases;
 	activated_neurons = functions::sigmoid(neurons_values);
 
}

void layer::update_weights(const Eigen::VectorXd& input, double rate) {
	weights -= rate * (input * delta.transpose());
	biases -= rate * delta;
}
