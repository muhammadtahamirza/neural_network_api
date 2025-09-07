#include "dataset.h"
#include "neural_network.h"



using namespace std;

int main3() {

	std::vector<Eigen::VectorXd> images;
	std::vector<Eigen::VectorXd> labels;


	std::vector<Eigen::VectorXd> test_images;
	std::vector<Eigen::VectorXd> test_labels;

	 

	std::vector<layer> layers;
	layers.emplace_back(784, 64); //hidden layer
	layers.emplace_back(64, 10); //ouput layer



	neural_network network(layers);
	cout << "layers connected to network successfully !!!\n\n";


	cout << "loading model ...\n";
	network.load_model("taha_model.bin");
	dataset::read_mnist_train_data("dataset/t10k-images.idx3-ubyte", test_images);
	dataset::read_mnist_test_label("dataset/t10k-labels.idx1-ubyte", test_labels);

	int ans = network.predict(test_images[10]);

	return 0;

	int menu = 0;
	while (menu != -1) {
		cout << endl;
		cout << "------------------------------\n";
		cout << "Select from menu : " << endl;
		cout << "	1. load training images" << endl;
		cout << "	2. load testing images" << endl;
		cout << "          ********    " << endl;
		cout << "	3. train and save model" << endl;
		cout << "	4. load and test model" << endl;
		cout << "	0. exit  program" << endl;
		cout << "------------------------------\n";
		cout << endl;
		cin >> menu;

		if (menu == 1)
		{
			dataset::read_mnist_train_data("dataset/train-images.idx3-ubyte", images);
			dataset::read_mnist_test_label("dataset/train-labels.idx1-ubyte", labels);
		}
		else if(menu==2){
			dataset::read_mnist_train_data("dataset/t10k-images.idx3-ubyte", test_images);
			dataset::read_mnist_test_label("dataset/t10k-labels.idx1-ubyte", test_labels);

		}
		else if (menu == 3) {
			cout << "Traingin started ...\n";

			network.train(images, labels, 0.1, 3);
			network.saveModel("taha_model.bin");


			cout << "ended" << endl;
		}
		else if (menu == 4) {
			cout << "loading model ...\n";
			network.load_model("taha_model.bin");
			network.test(test_images, test_labels);

		}
		else  {
			cout << "invalid try again ! " << endl;
		}
	}



	return 0;


	return 0;
}
