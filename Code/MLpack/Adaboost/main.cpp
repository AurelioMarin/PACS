#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/data/confusion_matrix.hpp>

#include <mlpack/methods/adaboost/adaboost.hpp>
#include <iostream>
#include <chrono>
#include <ctime>

using namespace mlpack;
using namespace mlpack::adaboost;
using namespace mlpack::tree;
using namespace mlpack::perceptron;

int main() {
    arma::mat InputData;
     arma::Row<size_t> Label;
     arma::mat TrainData;
     arma::Row<size_t> TrainLabel;
     arma::mat TestData;
     arma::Row<size_t> TestLabel;
     arma::mat probabilities;
     arma::Row<size_t> predictions;

     arma::mat Confusion;

     //DATA LOADING
     data::Load("../../Data/winequality-white.csv", InputData,false,true);
     Label=arma::conv_to<arma::Row<size_t>>::from( InputData.row(InputData.n_rows-1));
     InputData=InputData(arma::span(0,InputData.n_rows-2),arma::span::all);
     //DATA SPLIT
     double TestRatio;
     std::cout<<"Introduce the ratio of the Test set ";
     std::cin>>TestRatio;
     data::Split(InputData,Label,TrainData,TestData,TrainLabel,TestLabel,TestRatio);
     std::cout<<"Size Train"<<arma::size(InputData)<<" Size Label"<<arma::size(Label)<<std::endl;

     //Timer
     std::chrono::time_point<std::chrono::system_clock> start, end;
     start = std::chrono::system_clock::now();
     arma::mat inputData;

     //Machine learning algorithm
     Perceptron<> p(TrainData, TrainLabel.row(0), 10);
     AdaBoost<> adab(TrainData,TrainLabel.row(0),10,p);

     adab.Classify(TestData,predictions,probabilities);

     data::ConfusionMatrix(predictions, TestLabel, Confusion, 7);


     std::cout<<"Test result"<<std::endl;
     std::cout<<"Confusion matrix : "<<std::endl<<Confusion;

     // Stop the timer.
     end = std::chrono::system_clock::now();
     int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
             (end-start).count();

      std::cout << "Time required: " << elapsed_seconds << "microseconds\n";

     return 0;
}
