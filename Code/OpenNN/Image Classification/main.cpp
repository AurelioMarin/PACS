

#include <iostream>
#include <time.h>

#include "opennn/opennn.h"

using namespace OpenNN;

int main()
{        srand(static_cast<unsigned>(time(nullptr)));

        // Data set Reading

        DataSet data_set("../../Data/fashion-mnist_train.csv", ',', true);

        data_set.set_input();
        data_set.set_column_use(0,DataSet::VariableUse::Target);
        data_set.numeric_to_categorical(0);
        data_set.set_batch_instances_number(5);

        const Vector<size_t> inputs_dimensions({1, 28, 28});
        const Vector<size_t> targets_dimensions({10});
        data_set.set_input_variables_dimensions(inputs_dimensions);
        data_set.set_target_variables_dimensions(targets_dimensions);

        //Since it is an example, we are not going to train over the 60000 images
        const size_t total_instances = 125;
        data_set.set_instances_uses((Vector<string>(total_instances, "Training").assemble(Vector<string>(60000 - total_instances, "Unused"))));
        data_set.split_instances_random(0.8,0.1,0.1);

        // Neural network

        const size_t outputs_number = 10;

        NeuralNetwork neural_network;

        // Scaling layer

        ScalingLayer* scaling_layer = new ScalingLayer(inputs_dimensions);
        neural_network.add_layer(scaling_layer);

        const Vector<size_t> scaling_layer_outputs_dimensions = scaling_layer->get_outputs_dimensions();

        // Convolutional Block

        ConvolutionalLayer* convolutional_layer_1 = new ConvolutionalLayer(scaling_layer_outputs_dimensions, {8, 5, 5});
        neural_network.add_layer(convolutional_layer_1);

        const Vector<size_t> convolutional_layer_1_outputs_dimensions = convolutional_layer_1->get_outputs_dimensions();

        PoolingLayer* pooling_layer_1 = new PoolingLayer(convolutional_layer_1_outputs_dimensions);
        neural_network.add_layer(pooling_layer_1);

        const Vector<size_t> pooling_layer_1_outputs_dimensions = pooling_layer_1->get_outputs_dimensions();

        // Convolutional Block

        ConvolutionalLayer* convolutional_layer_2 = new ConvolutionalLayer(pooling_layer_1_outputs_dimensions, {4, 3, 3});
        neural_network.add_layer(convolutional_layer_2);

        const Vector<size_t> convolutional_layer_2_outputs_dimensions = convolutional_layer_2->get_outputs_dimensions();

        PoolingLayer* pooling_layer_2 = new PoolingLayer(convolutional_layer_2_outputs_dimensions);
        neural_network.add_layer(pooling_layer_2);

        const Vector<size_t> pooling_layer_2_outputs_dimensions = pooling_layer_2->get_outputs_dimensions();

        // Convolutional Block

        ConvolutionalLayer* convolutional_layer_3 = new ConvolutionalLayer(pooling_layer_2_outputs_dimensions, {2, 3, 3});
        neural_network.add_layer(convolutional_layer_3);

        const Vector<size_t> convolutional_layer_3_outputs_dimensions = convolutional_layer_3->get_outputs_dimensions();

        PoolingLayer* pooling_layer_3 = new PoolingLayer(convolutional_layer_3_outputs_dimensions);
        neural_network.add_layer(pooling_layer_3);

        const Vector<size_t> pooling_layer_3_outputs_dimensions = pooling_layer_3->get_outputs_dimensions();

        // Fully conected layer: Dense layer with number input equal to flatten

        PerceptronLayer* perceptron_layer = new PerceptronLayer(pooling_layer_3_outputs_dimensions.calculate_product(), 18);
        neural_network.add_layer(perceptron_layer);

        const size_t perceptron_layer_outputs = perceptron_layer->get_neurons_number();

        // Probabilistic layer

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer_outputs, outputs_number);
        neural_network.add_layer(probabilistic_layer);

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
        sgd_pointer->set_maximum_epochs_number(12);
        sgd_pointer->set_display_period(1);
        sgd_pointer->set_maximum_time(1800);

        const OptimizationAlgorithm::Results training_strategy_results = training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        Matrix<size_t> confusion = testing_analysis.calculate_confusion();

        cout << "\n\nConfusion matrix: \n" << endl << confusion << endl;
        cout << "\nAccuracy: " << (confusion.calculate_trace()/confusion.calculate_sum())*100 << " %" << endl << endl;

        // Save results

        data_set.save("data_set.xml");

        neural_network.save("neural_network.xml");

        training_strategy.save("training_strategy.xml");
        training_strategy_results.save("training_strategy_results.dat");

        return 0;

}
