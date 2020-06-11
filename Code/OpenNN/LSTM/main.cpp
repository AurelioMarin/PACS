

#include <iostream>
#include <time.h>

#include "opennn/opennn.h"

using namespace OpenNN;

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));

    // Data set Reading

    DataSet data_set;
    data_set.set_data_file_name("../../Data/shampo_sales.csv");
    data_set.set_separator(',');
    data_set.set_has_columns_names(true);
    size_t lags_number = 4;
    size_t steps_ahead = 1;

    data_set.set_lags_number(lags_number);
    data_set.set_steps_ahead_number(steps_ahead);

    data_set.set_time_index(0);

    data_set.set_missing_values_method("Mean");

    data_set.read_csv();


    const size_t inputs_number = data_set.get_input_variables_number();
    const size_t outputs_number = data_set.get_target_variables_number();
    const Vector<Descriptives> inputs_descriptives = data_set.scale_inputs_minimum_maximum();
    const Vector<Descriptives> targets_descriptives = data_set.scale_targets_minimum_maximum();

    data_set.split_instances_sequential();


    // Neural network

    NeuralNetwork neural_network;

    // Scaling layer

    ScalingLayer* scaling_layer_prt = new ScalingLayer(inputs_number);
    scaling_layer_prt->set_scaling_methods(ScalingLayer::NoScaling);

    neural_network.add_layer(scaling_layer_prt);

    const size_t scaling_layer_outputs_dimensions = scaling_layer_ptr->get_neurons_number();


    LongShortTermMemoryLayer* long_short_term_memory_layer_1_ptr = new LongShortTermMemoryLayer(scaling_layer_outputs_dimensions, 4);

    neural_network.add_layer(long_short_term_memory_layer_1_ptr);

    const size_t lstm_layer_1_outputs = long_short_term_memory_layer_1_ptr->get_neurons_number();

    PerceptronLayer* perceptron_layer_1_ptr = new PerceptronLayer(lstm_layer_1_outputs, 1);

    neural_network.add_layer(perceptron_layer_1_ptr);

    neural_network.print_summary();

    // Training strategy
    TrainingStrategy training_strategy(&neural_network, &data_set);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
    training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
    training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    AdaptiveMomentEstimation * 	 adam_ptr = training_strategy.get_adaptive_moment_estimation_pointer() ;

    adam_ptr->set_maximum_epochs_number(3000);
    adam_ptr->set_display_period(100);
    adam_ptr->set_maximum_time(1800);

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
