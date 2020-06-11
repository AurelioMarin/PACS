

#include <iostream>
#include <time.h>

#include "opennn/opennn.h"

using namespace OpenNN;

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));

    // Data set Reading

    DataSet data_set("../../Data/iris_data.csv",',',false);

    const size_t outputs_number = data_set.get_target_variables_number();
    const size_t inputs_number = data_set.get_input_variables_number();
    const Vector<Descriptives> inputs_descriptives = data_set.scale_inputs_minimum_maximum();
    const Vector<Descriptives> targets_descriptives = data_set.scale_targets_minimum_maximum();
    data_set.set_batch_instances_number(5);
    data_set.split_instances_random();
    cout<<outputs_number<<" "<<inputs_number<<endl;

    // Neural network
    NeuralNetwork neural_network;

    // Scaling layer

    ScalingLayer* scaling_layer = new ScalingLayer(inputs_number);
    scaling_layer->set_descriptives(inputs_descriptives);
    scaling_layer->set_scaling_methods(ScalingLayer::MeanStandardDeviation);

    neural_network.add_layer(scaling_layer);

    const size_t scaling_layer_outputs_dimensions = scaling_layer->get_neurons_number();
    //Perceptron block

    PerceptronLayer* perceptron_layer_1 = new PerceptronLayer(scaling_layer_outputs_dimensions,8);
    perceptron_layer_1->set_activation_function(PerceptronLayer::RectifiedLinear);
    neural_network.add_layer(perceptron_layer_1);

    const size_t perceptron_layer_1_outputs = perceptron_layer_1->get_neurons_number();

    ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer_1_outputs, outputs_number);
    probabilistic_layer->set_activation_function(ProbabilisticLayer::ActivationFunction::Softmax);

    neural_network.add_layer(probabilistic_layer);

    neural_network.print_summary();

    // Training strategy
    TrainingStrategy training_strategy(&neural_network, &data_set);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
    training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
    training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

    QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

    quasi_Newton_method_pointer->set_minimum_loss_decrease(1.0e-6);

    quasi_Newton_method_pointer->set_loss_goal(1.0e-3);

    quasi_Newton_method_pointer->set_minimum_parameters_increment_norm(0.0);

    quasi_Newton_method_pointer->perform_training();



    const OptimizationAlgorithm::Results training_strategy_results = training_strategy.perform_training();
        // Testing analysis
    TestingAnalysis testing_analysis(&neural_network, &data_set);
    cout<<endl<<"Testing Analysis"<<endl;
    const Matrix<size_t> confusion = testing_analysis.calculate_confusion();

    Vector<double> multiple_classification_tests = testing_analysis.calculate_multiple_classification_tests();

    cout << "Confusion: " << endl;
    cout << confusion << endl;

    // Save results

    data_set.save("data_set.xml");

    neural_network.save("neural_network.xml");

    training_strategy.save("training_strategy.xml");
    training_strategy_results.save("training_strategy_results.dat");
    confusion.save_csv("ConfusionMatrix.csv");

    return 0;

}
