#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/data/confusion_matrix.hpp>

#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <iostream>
#include <chrono>
#include <ctime>

using namespace mlpack;
using namespace mlpack::regression;
int main() {
    arma::mat InputData;
    arma::Row<size_t> Label;
    arma::mat TrainData;
    arma::Row<size_t> TrainLabel;
    arma::mat TestData;
    arma::Row<size_t> TestLabel;
    arma::rowvec parameters;
    arma::Row<size_t> predictions;

    double train_error;
    double test_error;
    double train_accu;
    double test_accu;
    arma::mat Confusion;

    //DATA LOADING

    data::Load("../../Data/data_banknote_authentication.txt", InputData,false,true);
    Label=arma::conv_to<arma::Row<size_t>>::from( InputData.row(InputData.n_rows-1));
    InputData=InputData(arma::span(0,InputData.n_rows-2),arma::span::all);
    //DATA SPLIT
    double TestRatio;
    std::cout<<"Introduce the ratio of the Test set ";
    std::cin>>TestRatio;
    data::Split(InputData,Label,TrainData,TestData,TrainLabel,TestLabel,TestRatio);
    std::cout<<"Size Train"<<arma::size(TrainData.t())<<" Size Test"<<arma::size(TestData)<<std::endl;



     //Timer setting
     std::chrono::time_point<std::chrono::system_clock> start, end;
     start = std::chrono::system_clock::now();
     // Logistic regressor
     LogisticRegression lr(TrainData,TrainLabel);

     parameters=lr.Parameters();
     std::cout<<"Parameters obtained"<<std::endl;
     std::cout<<parameters<<std::endl;

     train_error=lr.ComputeError(TrainData,TrainLabel);
     train_accu=lr.ComputeAccuracy(TrainData,TrainLabel);
     test_error=lr.ComputeError(TestData,TestLabel);
     test_accu=lr.ComputeAccuracy(TestData,TestLabel);
     lr.Classify(TestData,predictions);

    data::ConfusionMatrix(predictions, TestLabel, Confusion, 2);

     std::cout<<"Training result"<<std::endl;
     std::cout<<"Error : "<<train_error<<std::endl;
     std::cout<<"Accuracy : "<<train_accu<<std::endl;
     std::cout<<"Test result"<<std::endl;
     std::cout<<"Error : "<<test_error<<std::endl;
     std::cout<<"Accuracy : "<<test_accu<<std::endl;
     std::cout<<"Confusion matrix : "<<std::endl<<Confusion;



     // Stop the timer.
     end = std::chrono::system_clock::now();
     int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
             (end-start).count();



      std::cout << "Time required: " << elapsed_seconds << "microseconds\n";


     return 0;
}
