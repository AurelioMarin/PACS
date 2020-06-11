#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron/perceptron.hpp>
#include <mlpack/methods/perceptron/learning_policies/simple_weight_update.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/data/confusion_matrix.hpp>


#include <iostream>
#include <chrono>
#include <ctime>
using namespace mlpack;
using namespace mlpack::perceptron;
int main() {
    arma::mat InputData;
    arma::Row<size_t> Label;
    arma::mat TrainData;
    arma::Row<size_t> TrainLabel;
    arma::mat TestData;
    arma::Row<size_t> TestLabel;// Load into this matrix.

    data::Load("../../Data/iris_data.csv", InputData,false,true);
    InputData=InputData(arma::span::all,arma::span(0,InputData.n_cols-2));

    data::Load("../../Data/iris_labels.csv", Label,false, true);
    //DATA SPLIT
    float TestRatio;
    std::cout<<"Introduce the ratio of the Test set ";
    std::cin>>TestRatio;
    data::Split(InputData,Label,TrainData,TestData,TrainLabel,TestLabel,TestRatio);

    size_t classes;
    std::cout<<"Introduce the number of classes  ";
    std::cin>>classes;

    arma::Row<size_t> predictedLabels(TestData.n_cols);
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    Perceptron<> model(TrainData,TrainLabel.row(0),classes,1000);

    model.Classify(TestData, predictedLabels);

    end = std::chrono::system_clock::now();
    int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
            (end-start).count();
    arma::mat Confusion;
    data::ConfusionMatrix(TestLabel, predictedLabels, Confusion, 3);
    std::cout<<"Confusion Matrix:  "<<std::endl;
    std::cout<<Confusion<<std::endl;

    Confusion.save("confusion.txt", arma::raw_ascii);
    std::cout << "Time required: " << elapsed_seconds << "microseconds\n";

    return 0;
}
