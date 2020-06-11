#include <mlpack/core.hpp>
#include <mlpack/methods/pca/pca.hpp>
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>
#include <iostream>
#include <chrono>
#include <ctime>
using namespace mlpack;
using namespace mlpack::pca;
int main() {
    arma::mat InputData;
        data::Load("../../Data/iris_values.csv", InputData);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    // PCA
    PCA<> p;
    p.Apply(InputData,2);
    // Stop the timer.
    end = std::chrono::system_clock::now();
    int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
            (end-start).count();

    std::cout << "Time required: " << elapsed_seconds << "microseconds\n";
    std::cout<< "Size of the transformed data"<<size(InputData)<<std::endl;

    InputData.save("TransformedData.csv", arma::csv_ascii);

    return 0;
}
