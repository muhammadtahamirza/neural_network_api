#pragma once

#include <iostream>
#include<vector>
#include <fstream>
 #include <cstdint>

 
#include <Eigen/Dense>

namespace dataset {
    void read_mnist_train_data(const std::string& path,
        std::vector< Eigen::VectorXd>& data);

    void read_mnist_train_label(const std::string& path,
        std::vector<Eigen::VectorXd>& data);  

    void read_mnist_test_data(const std::string& path,
        std::vector<Eigen::VectorXd>& data);


    void read_mnist_test_label(const std::string& path,
        std::vector<Eigen::VectorXd>& data);

}
