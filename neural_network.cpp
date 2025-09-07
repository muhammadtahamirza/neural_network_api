#include "neural_network.h"
#include <fstream>




void neural_network::load_model(const std::string& filename) {

	std::ifstream infile(filename, std::ios::in | std::ios::binary);
	std::cout << "opening file for reading model ...\n";
	if (!infile.is_open()) {
		std::cerr << "Info: No saved model found: " << filename << ". Starting from scratch." << std::endl;
		return ;
	}
	std::cout << "Starging Reading data from file : " << filename << " ..." << std::endl;


	size_t num_layers;
	infile.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

	std::vector<Eigen::VectorXd> biasVectors(num_layers);
	std::vector<Eigen::MatrixXd> weightMatrices(num_layers);

	// 3. For each layer, read its dimensions, resize the Eigen objects, and read the data
	for (size_t i = 0; i < num_layers; ++i) {
		// Read the dimensions for the weight matrix
		Eigen::Index rows, cols;
		infile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
		infile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

		// Resize the matrix and read the data directly into it
		weightMatrices[i].resize(rows, cols);
		infile.read(reinterpret_cast<char*>(weightMatrices[i].data()), rows * cols * sizeof(double));

		// Read the size for the bias vector
		Eigen::Index bias_size;
		infile.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));

		// Resize the vector and read the data directly into it
		biasVectors[i].resize(bias_size);
		infile.read(reinterpret_cast<char*>(biasVectors[i].data()), bias_size * sizeof(double));
	}
	infile.close();
	std::cout << "reading complete successfully ...\n";

	int total_layers = layers.size();
	std::cout << "loading model in network.....\n ";
	for (int layer = 0; layer < total_layers; layer++)
	{
		layers[layer].biases = biasVectors[layer];
		layers[layer].weights = weightMatrices[layer];
	}
	std::cout << " ********  model loaded successfuly *********\n\n";
}








void neural_network::forward(Eigen::VectorXd& input) {
 	layers[0].forward(input);
	for (int i = 1; i < layers.size(); i++)
	{
 		layers[i].forward(layers[i - 1].activated_neurons);

	}

}

Eigen::VectorXd neural_network::get_output() {
	return layers.back().activated_neurons;
}

void neural_network::backward_and_update(Eigen::VectorXd& input, Eigen::VectorXd& target, double rate) {  //just calculating the delta values

	// for oouput layer
	Eigen::VectorXd error_output = functions::error_function_derivative(get_output(), target); //dc/da
	Eigen::VectorXd sigmoid_derivative = functions::sigmoid_derivative(layers.back().neurons_values);

	Eigen::VectorXd delta = error_output.cwiseProduct(sigmoid_derivative);
	layers.back().delta = delta;



	//for hidden layer will use ouput delta for chaining
	int second_last_layer = layers.size() - 2;
	for (int i = second_last_layer; i >= 0; i--)
	{

		Eigen::VectorXd hidden_erorr = layers[i + 1].weights * layers[i + 1].delta;  //for chain rule   matrix weight * vector delta for the prev dc/da

		Eigen::VectorXd sigmoid_derivative_hidden = functions::sigmoid_derivative(layers[i].neurons_values);
		Eigen::VectorXd delta_hidden = hidden_erorr.cwiseProduct(sigmoid_derivative_hidden);
		layers[i].delta = delta_hidden;

	}

	layers[0].update_weights(input, rate);  //actually updating the weights of alllaers layers
	for (int i = 1; i < layers.size(); i++)
	{
		auto& prev_layer = layers[i - 1];
 
		layers[i].update_weights(prev_layer.activated_neurons, rate);
 	}

 
}


void neural_network::train(std::vector<Eigen::VectorXd>& images, std::vector<Eigen::VectorXd>& labels, double rate, int epochs) {
	if (images.size() != labels.size()) {
		std::cout << "not equal" << std::endl;
		throw "labels and images must be equal";
	}

	double loss = 0;
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		std::cout << "********    Epoch : " << epoch + 1 << " training ...     ***********\n\n";
		int total_images = images.size();
		int count = 1;
		std::cout << "stargin trainging images ....\n";

		for (int image = 0; image < total_images; image++)
		{
 			forward(images[image]); //calculating activations

 			loss += functions::error_function(layers.back().activated_neurons, labels[image]);

 			backward_and_update(images[image], labels[image], rate);

			if (image !=0 &&image % 10000 == 0) {
				std::cout <<"  -> " << image << " images Trained in model successsfully !!\nLoss : "<<loss/double(image)<<"\n\n";
			}
		}
		std::cout << "*************\n";
		std::cout << "  ->  60K images Trained in model successsfully !!";
		std::cout << "*************\n";

		loss /= double(total_images);

		std::cout << "Epoch :  " << epoch + 1<<  "  Loss: " << loss << std::endl;


	}
}





void neural_network::saveModel(	const std::string& filename) {


	std::ofstream outfile(filename, std::ios::out | std::ios::binary);
	std::cout << "opeing file for saving model ...\n";
	if (!outfile.is_open()) {
		throw std::runtime_error("Could not open file for saving: " + filename);
	}

	std::cout << "Starting Writing data in file : " << filename << " ..." << std::endl;
	// 1. Write the number of layers (i.e., the size of the vector)

	size_t num_layers = layers.size();
	outfile.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

	// 2. For each layer, write its dimensions and data
	for (size_t i = 0; i < num_layers; ++i) {
		// Write the dimensions of the weight matrix (rows, cols)
		Eigen::Index rows = layers[i].weights.rows();
		Eigen::Index cols = layers[i].weights.cols();
		outfile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
		outfile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

		outfile.write(reinterpret_cast<const char*>(layers[i].weights.data()), rows * cols * sizeof(double));

		// Write the size of the bias vector
		Eigen::Index bias_size = layers[i].biases.size();
		outfile.write(reinterpret_cast<const char*>(&bias_size), sizeof(bias_size));

		// Write the raw data of the bias vector
		outfile.write(reinterpret_cast<const char*>(layers[i].biases.data()), bias_size * sizeof(double));
	}

	outfile.close();
	std::cout << "Model saved to " << filename << " sucessfylly !!!" << std::endl;
}





void neural_network::test(std::vector<Eigen::VectorXd>& inputs, std::vector<Eigen::VectorXd>& target_outputs)
{

	int correct_predictions = 0;
	std::cout << "tesitng model ...\n";
	int total = inputs.size();
	for (size_t i = 0; i < inputs.size(); ++i) {
		Eigen::VectorXd input = inputs[i];
		Eigen::VectorXd target_output = target_outputs[i];

		forward(input);

		Eigen::VectorXd output = get_output();

		int predicted_index;
		int target_index;
		output.maxCoeff(&predicted_index);
		target_output.maxCoeff(&target_index);

		if (predicted_index == target_index) {
			correct_predictions++;
		}

		if (i  ==total/2 ) {
			std::cout << "Comparing ...\n";
			for (int i = 0; i < output.size(); i++)
			{
				std::cout << output[i] << " :  " << target_output[i] << std::endl;
			}
			std::cout << "coparing done !!\n\n";
		}

	}
	std::cout << "testing completed succesfully !!!\n";
	double accuracy = static_cast<double>(correct_predictions) / inputs.size();
	std::cout << "Test Accuracy: " << accuracy*100.0 <<" %" << std::endl;
}

int neural_network::predict(Eigen::VectorXd& input) {
	std::cout << "Starting prediction ...\n";
	std::cout << "forwarding in layers with trained weigths and biased ...\n";
	forward(input);
	std::cout << "Output laeyr created successfully ...\n";

	Eigen::VectorXd output = get_output();
	int result = -1;
	double max = -9999999999;

	for (int i = 0; i < output.size(); i++)
	{
		if (output[i]>max) {
			max = output[i];
			result = i;
		}
	}
	std::cout << "prediction completed !!!\n";
	return result;
}