
#pragma once
#include "layer.h"
#include<iostream>
 #include<vector>


class neural_network
{
public:
	std::vector<layer> layers;

	neural_network(std::vector<layer> layers) : layers(layers) {}

	void load_model(const std::string& filename);
	void saveModel(const std::string& filename);
	void forward(Eigen::VectorXd& input);

	Eigen::VectorXd get_output();

	void backward_and_update(Eigen::VectorXd& input, Eigen::VectorXd& target, double rate); 


	void train(std::vector<Eigen::VectorXd>& images, std::vector<Eigen::VectorXd>& labels, double rate, int epochs);
	void test(std::vector<Eigen::VectorXd>& inputs, std::vector<Eigen::VectorXd>& target_outputs);

	int predict(Eigen::VectorXd & input);
};

