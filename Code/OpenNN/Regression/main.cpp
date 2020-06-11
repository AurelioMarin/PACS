

#include <iostream>
#include <time.h>

#include "opennn/opennn.h"

using namespace OpenNN;

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));

    // Data set Reading
    DataSet data_set("../../Data/Boston.csv",',',true);
    data_set.set_columns_uses({"UnusedVariable","Input","Input","Input","Input","Input","Input","Input","Input","Input","Input","Input","Input","Input","Target"});

    const size_t inputs_number = data_set.get_input_variables_number();
    const size_t outputs_number = data_set.get_target_variables_number();
    const Vector<Descriptives> inputs_descriptives = data_set.scale_inputs_minimum_maximum();
    const Vector<Descriptives> targets_descriptives = data_set.scale_targets_minimum_maximum();

    data_set.split_instances_random();


    // Neural network

    NeuralNetwork neural_network;

    // Scaling layer

    ScalingLayer* scaling_layer_ptr = new ScalingLayer(inputs_number);
    scaling_layer_ptr->set_descriptives(inputs_descriptives);

    neural_network.add_layer(scaling_layer_ptr);

    const size_t scaling_layer_outputs_dimensions = scaling_layer_ptr->get_neurons_number();
    //Perceptron block

    PerceptronLayer* perceptron_layer_1_ptr = new PerceptronLayer(scaling_layer_outputs_dimensions,64);
    perceptron_layer_1_ptr->set_activation_function(PerceptronLayer::RectifiedLinear);
    neural_network.add_layer(perceptron_layer_1_ptr);

    const size_t perceptron_layer_1_outputs = perceptron_layer_1_ptr->get_neurons_number();

    PerceptronLayer* perceptron_layer_2_ptr = new PerceptronLayer(perceptron_layer_1_outputs,64);
    perceptron_layer_2_ptr->set_activation_function(PerceptronLayer::RectifiedLinear);
    neural_network.add_layer(perceptron_layer_2_ptr);

    const size_t perceptron_layer_2_outputs = perceptron_layer_2_ptr->get_neurons_number();

    PerceptronLayer* perceptron_layer_3_ptr = new PerceptronLayer(perceptron_layer_2_outputs,1);
    neural_network.add_layer(perceptron_layer_3_ptr);

    const size_t perceptron_layer_3_outputs = perceptron_layer_3_ptr->get_neurons_number();

    UnscalingLayer* unscaling_layer_ptr = new UnscalingLayer(perceptron_layer_3_outputs);
    neural_network.add_layer(unscaling_layer_ptr);

    neural_network.print_summary();

    // Training strategy
    TrainingStrategy training_strategy(&neural_network, &data_set);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
    training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    StochasticGradientDescent* sgd_pointer = training_strategy.get_stochastic_gradient_descent_pointer();

    sgd_pointer->set_initial_learning_rate(1.0e-3);
    sgd_pointer->set_momentum(0.9);
    sgd_pointer->set_minimum_loss_increase(1.0e-6);
    sgd_pointer->set_maximum_epochs_number(1000);
    sgd_pointer->set_display_period(100);
    sgd_pointer->set_maximum_time(1800);

    const OptimizationAlgorithm::Results training_strategy_results = training_strategy.perform_training();


    // Testing analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);
    cout<<endl<<"Testing Analysis"<<endl;
    Vector< double > TestError=testing_analysis.calculate_testing_errors();
    cout<<"Sum Squared error  :"<<TestError[0]<<endl;
    cout<<"Mean Squared error   :"<<TestError[1]<<endl;
    cout<<"Root Mean Squared error   :"<<TestError[2]<<endl;
    cout<<"Normalized Squared error   :"<<TestError[3]<<endl;

    const TestingAnalysis::LinearRegressionAnalysis linear_regression_results = testing_analysis.perform_linear_regression_analysis()[0];
    cout << "Linear Regression analysis"<<endl;
    cout<<"Correlation    : " << linear_regression_results.correlation << endl;

    // Save results

    data_set.save("data_set.xml");

    neural_network.save("neural_network.xml");

    training_strategy.save("training_strategy.xml");
    training_strategy_results.save("training_strategy_results.dat");
    linear_regression_results.save("linear_regression_analysis_results.dat");
    TestError.save("Error.dat");

    return 0;

}
