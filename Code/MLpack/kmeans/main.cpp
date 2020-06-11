#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/core/data/confusion_matrix.hpp>

#include <mlpack/methods/kmeans/kmeans.hpp>
#include <iostream>
#include <chrono>
#include <ctime>

using namespace mlpack;
using namespace mlpack::kmeans;
int main() {
    arma::mat InputData;
    arma::Row<size_t> Label;
    arma::mat TrainData;
    arma::Row<size_t> TrainLabel;
    arma::mat TestData;
    arma::Row<size_t> TestLabel;// Load into this matrix.

    data::Load("../../Data/iris_values.csv", InputData);
    data::Load("../../Data/iris_labels.csv", Label);
    //DATA SPLIT
    double TestRatio;
    std::cout<<"Introduce the ratio of the Test set ";
    std::cin>>TestRatio;
    data::Split(InputData,Label,TrainData,TestData,TrainLabel,TestLabel,TestRatio);

    //Setting
    size_t clusters;
    std::cout<<"Introduce the number of cluster to search ";
    std::cin>>clusters;

    arma::Row<size_t> assignments;
    arma::mat centroids;
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    // KMeans definition
    KMeans<> k;
    k.Cluster(TrainData, clusters, assignments, centroids);

    // Stop the timer.
    end = std::chrono::system_clock::now();
    int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
            (end-start).count();

    std::cout << "Time required: " << elapsed_seconds << "microseconds\n";

    arma::mat Confusion;
    data::ConfusionMatrix(assignments, TrainLabel, Confusion, 3);

    Confusion.save("confusion.txt", arma::raw_ascii);
    assignments.save("labels.txt", arma::raw_ascii);
    centroids.save("centroids.txt", arma::raw_ascii);
    return 0;
}
