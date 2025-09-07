#include "functions.h"

#include<vector>
#include<fstream>
#include<iostream>

namespace functions {
    Eigen::VectorXd sigmoid(const Eigen::VectorXd& v) {
        return 1.0 / (1.0 + (-v.array()).exp());
    }

    Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& v) {
        return sigmoid(v).array() * (1.0 - sigmoid(v).array());
    }

    double error_function(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
        return 0.5 * (output - target).squaredNorm();
    }

    Eigen::VectorXd error_function_derivative(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
        return output - target;
    }

 

    


}
