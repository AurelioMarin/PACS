

#include <iostream>
#include <time.h>

#include "opennn/opennn.h"

using namespace OpenNN;

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));

    // Data set Reading

    DataSet data_set("../../Data/data_banknote_authentication.txt",',',false);

    data_set.split_instances_random();

    const Vector<string> inputs_names = data_set.get_input_variables_names();
    const Vector<string> targets_names = data_set.get_target_variables_names();

    const Vector<Descriptives> inputs_descriptives = data_set.scale_inputs_minimum_maximum();

    // Neural network
    NeuralNetwork neural_network(NeuralNetwork::Classification, {8, 6, 1});

    // Scaling layer

    ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
    scaling_layer_pointer->set_descriptives(inputs_descriptives);
    scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);

    neural_network.print_summary();

    // Training strategy
    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
    training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::RegularizationMethod::L2);
    training_strategy.get_loss_index_pointer()->set_regularization_weight(0.001);

    AdaptiveMomentEstimation * 	 adam_pointer = training_strategy.get_adaptive_moment_estimation_pointer() ;

    adam_pointer->set_maximum_epochs_number(3000);
    adam_pointer->set_display_period(100);
    adam_pointer->set_maximum_time(1800);



    const OptimizationAlgorithm::Results training_strategy_results = training_strategy.perform_training();
//Model Selection
    ModelSelection model_selection(&training_strategy);

    model_selection.perform_neurons_selection();
        // Testing analysis
    data_set.unscale_inputs_minimum_maximum(inputs_descriptives);

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    Matrix<size_t> confusion = testing_analysis.calculate_confusion();

    Vector<double> binary_classification_tests = testing_analysis.calculate_binary_classification_tests();
    cout << endl<<"Test Analysis"<< endl;
    cout << "Confusion:"<< endl << confusion << endl;

    cout << "Binary classification tests: " << endl;
    cout << "Classification accuracy         : " << binary_classification_tests[0] << endl;
    cout << "Error rate                      : " << binary_classification_tests[1] << endl;
    cout << "Sensitivity                     : " << binary_classification_tests[2] << endl;
    cout << "Specificity                     : " << binary_classification_tests[3] << endl;
    cout << "Precision                       : " << binary_classification_tests[4] << endl;
    cout << "Positive likelihood             : " << binary_classification_tests[5] << endl;
    cout << "Negative likelihood             : " << binary_classification_tests[6] << endl;
    cout << "F1 score                        : " << binary_classification_tests[7] << endl;
    cout << "False positive rate             : " << binary_classification_tests[8] << endl;
    cout << "False discovery rate            : " << binary_classification_tests[9] << endl;
    cout << "False negative rate             : " << binary_classification_tests[10] << endl;
    cout << "Negative predictive value       : " << binary_classification_tests[11] << endl;
    cout << "Matthews correlation coefficient: " << binary_classification_tests[12] << endl;
    cout << "Informedness                    : " << binary_classification_tests[13] << endl;
    cout << "Markedness                      : " << binary_classification_tests[14] << endl;
    // Save results

    data_set.save("../data/data_set.xml");

    neural_network.save("../data/neural_network.xml");
    neural_network.save_expression("../data/expression.txt");

    training_strategy.save("../data/training_strategy.xml");
//        training_strategy_results.save("../data/training_strategy_results.dat");

    confusion.save_csv("../data/confusion.csv");
//        binary_classification_tests.save("../data/binary_classification_tests.dat");

    /*
    data_set.save("data_set.xml");

    neural_network.save("neural_network.xml");

    training_strategy.save("training_strategy.xml");
    training_strategy_results.save("training_strategy_results.dat");
    confusion.save_csv("ConfusionMatrix.csv");
*/
    return 0;

}
