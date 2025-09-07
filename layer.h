#include"functions.h"




class layer {
public:

	Eigen::MatrixXd weights; // weights per neruon
	Eigen::VectorXd biases;
	Eigen::VectorXd neurons_values;
	Eigen::VectorXd activated_neurons;

	Eigen::VectorXd delta;

	layer(int inputs, int neurons_count);

	void forward(const Eigen::VectorXd& input);

	void update_weights(const Eigen::VectorXd& input, double rate);

	
};


