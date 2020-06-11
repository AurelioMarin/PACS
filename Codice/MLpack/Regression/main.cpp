#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>

#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

#include <iostream>
#include <chrono>
#include <ctime>

using namespace mlpack;
using namespace mlpack::regression;
int main() {
    arma::mat predictors;
    arma::rowvec responses;
    arma::mat traindata;
    arma::rowvec trainresponses;
    arma::mat testdata;
    arma::rowvec testresponses;
    arma::vec predictions;
    arma::vec parameters;
    arma::rowvec beta;

    double train_error_linear;
    double test_error_linear;
    double train_error_lar;
    double test_error_lar;

    data::Load("../../Data/Swedish_auto.txt", predictors);
    responses=arma::conv_to<arma::rowvec>::from( predictors.row(predictors.n_rows-1));
    predictors=predictors(0,arma::span::all);

   double testratio;
     std::cout<<"Introduce the ratio of the Test set ";
     std::cin>>testratio;
     data::Split(predictors,responses,traindata,testdata,trainresponses,testresponses,testratio);
     std::cout<<"Size Train"<<arma::size(traindata)<<" Size Test"<<arma::size(testdata)<<std::endl;

    std::cout<<"Begin linear regression algorithm"<<std::endl;
   std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    // Do some things.
    LinearRegression lr(traindata, trainresponses,0.2,true );

    parameters=lr.Parameters();
    std::cout<<"Parameters obtained"<<std::endl;
    std::cout<<parameters;
    if (lr.Intercept()){
        std::cout<<"Be aware that the models includes intercept"<<std::endl;
    }    else{
        std::cout<<"Be aware that the models does not include intercept"<<std::endl;
    }

    train_error_linear=lr.ComputeError(traindata, trainresponses);
    test_error_linear=lr.ComputeError(testdata, testresponses);


    std::cout<<"Training Regression L2 square error  :"<<train_error_linear<<std::endl;
    std::cout<<"Test Regression L2 square error  :"<<test_error_linear<<std::endl;
    // Stop the timer.
    end = std::chrono::system_clock::now();
    int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
            (end-start).count();

    std::cout << "Time required: " << elapsed_seconds << "microseconds\n";

    std::cout<<"Begin least-angle regression algorithm"<<std::endl;
    start = std::chrono::system_clock::now();
    // Do some things.
    LARS lars(traindata, trainresponses);

    beta=lars.Beta();
    std::cout<<"Beta obtained"<<std::endl;
    std::cout<<beta;

    train_error_lar=lars.ComputeError(traindata, trainresponses);
    train_error_lar=train_error_lar/traindata.n_cols;
    test_error_lar=lars.ComputeError(testdata, testresponses);
    test_error_lar=test_error_lar/testdata.n_cols;


    std::cout<<"Training Regression L2 square error  :"<<train_error_lar<<std::endl;
    std::cout<<"Test Regression L2 square error  :"<<test_error_lar<<std::endl;
    // Stop the timer.
    end = std::chrono::system_clock::now();
    int elapsed_seconds2 = std::chrono::duration_cast<std::chrono::microseconds>
            (end-start).count();

    std::cout << "Time required: " << elapsed_seconds2 << "microseconds\n";


    return 0;
}
//g++ -std=c++17 DataSplit.cpp main.cpp -o main -I/home/aumar/PACS/Proyecto/mlpack-3.3.1/build/include -I/usr/include -larmadillo -lboost_serialization -lmlpack -lboost_program_options   -fopenmp