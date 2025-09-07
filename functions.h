#pragma once


#include <Eigen/Dense>



namespace functions {


    Eigen::VectorXd sigmoid(const Eigen::VectorXd& v);
    Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& v);
    double error_function(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
    Eigen::VectorXd error_function_derivative(const Eigen::VectorXd& output, const Eigen::VectorXd& target);


 }
