#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "../decision_tree/utility/random.hpp"
#include "../decision_tree/core/splitter.hpp"

namespace {

using ::testing::DoubleLE;

auto calculate_num_classes_list = [](const std::vector<std::vector<std::string>>& classes) {
    std::vector<unsigned long> num_classes_list(classes.size(), 0);
    for (std::size_t o=0; o<classes.size(); o++) {
        num_classes_list[o] = classes[o].size();
    }
    return num_classes_list;
};

auto init_class_weight = [](unsigned long num_outputs, 
                            unsigned long num_samples, 
                            unsigned long max_num_classes, 
                            const std::vector<long>& label,
                            const std::vector<unsigned long>& num_classes_list) {

    std::vector<double>  class_weight(num_outputs * max_num_classes, 1.0);
    for (unsigned long o=0; o<num_outputs; ++o) { // process each output independently
        std::vector<long> bincount(num_classes_list[o], 0);
        for (unsigned long i = 0; i < num_samples; ++i) {
            bincount[label[i*num_outputs+o]]++;
        }
        for (unsigned long c = 0; c < num_classes_list[o]; ++c) {
            class_weight[o*max_num_classes + c] =
                    (static_cast<double>(num_samples) / bincount[c]) / num_classes_list[o];
        }
    }

    return class_weight;
};

class SplitterTest : public ::testing::Test{    
public:
    virtual void SetUp() {
        std::vector<std::vector<std::string>> classes = {{"setosa", "versicolor", "virginica"}};
        std::vector<std::string> features = {"sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"};
        std::vector<double> X = {5.2, 3.3, 1.2, 0.3,
                                4.8, 3.1 , 1.6, 0.2,
                                4.75, 3.1, 1.32, 0.1,
                                5.9, 2.6, 4.1, 1.2,
                                5.1, 2.2, 3.3, 1.1,
                                5.2, 2.7, 4.1, 1.3,
                                6.6, 3.1, 5.25, 2.2,
                                6.3, 2.5, 5.1, 2.0,
                                6.5, 3.1, 5.2, 2.1};
        std::vector<long> y = {0, 0, 0, 1, 1, 1, 2, 2, 2};

        std::vector<unsigned long> num_classes_list = calculate_num_classes_list(classes);
        unsigned long num_outputs = classes.size();
        unsigned long num_samples = y.size() / num_outputs;
        unsigned long num_features = features.size();
        unsigned long max_num_features = 0;
        unsigned long max_num_classes = *std::max_element(begin(num_classes_list),
                                                        end(num_classes_list));

        std::vector<double>  class_weight = init_class_weight(num_outputs, 
                                                            num_samples, 
                                                            max_num_classes,
                                                            y, 
                                                            num_classes_list);
        decisiontree::RandomState random_state;
        splitter = new decisiontree::Splitter(num_outputs,
                                              num_samples,
                                              num_features, 
                                              max_num_features,
                                              max_num_classes,
                                              class_weight,
                                              num_classes_list,                                               
                                              "gini", 
                                              "best",
                                              random_state);
    }

    virtual void TearDown() {
        if (splitter) {
            delete splitter;
            splitter = nullptr;
        }
    }

    decisiontree::Splitter* splitter;
};

TEST_F(SplitterTest, InitNodeTest) {
    std::vector<double> X = {5.2, 3.3, 1.2, 0.3,
                                4.8, 3.1 , 1.6, 0.2,
                                4.75, 3.1, 1.32, 0.1,
                                5.9, 2.6, 4.1, 1.2,
                                5.1, 2.2, 3.3, 1.1,
                                5.2, 2.7, 4.1, 1.3,
                                6.6, 3.1, 5.25, 2.2,
                                6.3, 2.5, 5.1, 2.0,
                                6.5, 3.1, 5.2, 2.1};
    std::vector<long> y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    unsigned long num_samples = y.size();

    unsigned long feature_index = 0;
    unsigned long partition_index = 0;
    double partition_threshold = std::numeric_limits<double>::quiet_NaN();
    double improvement = 0.0;
    int has_missing_value = -1;
    
    splitter->init_node(y, 0, num_samples);
    double impurity = splitter->criterion_ptr_->get_node_impurity();

    std::vector<std::vector<HistogramType>> node_weighted_histogram = splitter->criterion_ptr_->get_node_weighted_histogram();
    std::vector<std::vector<double>> expect = {{3.0, 3.0, 3.0}};
    EXPECT_THAT(node_weighted_histogram, ::testing::ContainerEq(expect));
    EXPECT_PRED_FORMAT2(DoubleLE, impurity, double(2.0/3.0));

    splitter->split_node(X, y, feature_index, partition_index, partition_threshold, improvement, has_missing_value);
    
    std::cout << "feature_index= " << feature_index << std::endl;
    std::cout << "improvement = " << improvement << std::endl;
    std::cout << "partition_index = " << partition_index << std::endl;
    std::cout << "partition_threshold = " << partition_threshold << std::endl;
    std::cout << "has_missing_value = " << has_missing_value << std::endl;

} 


} // namespace
