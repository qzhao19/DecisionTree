#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "decision_tree/core/criterion/gini.hpp"

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

class GiniCriterionTest : public ::testing::Test{    
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
        unsigned long max_num_classes = *std::max_element(begin(num_classes_list),
                                                        end(num_classes_list));

        std::vector<double>  class_weight = init_class_weight(num_outputs, 
                                                            num_samples, 
                                                            max_num_classes,
                                                            y, 
                                                            num_classes_list);
        gini = new decisiontree::Gini(num_outputs, 
                                    num_samples, 
                                    max_num_classes, 
                                    num_classes_list, 
                                    class_weight);
    }

    virtual void TearDown() {
        if (gini) {
            delete gini;
            gini = nullptr;
        }
    }

    decisiontree::Gini* gini;
};

TEST_F(GiniCriterionTest, ComputeNodeHistogramTest) {
    std::vector<long> y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    unsigned long num_samples = y.size();
    std::vector<unsigned long> sample_indices(num_samples);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);

    gini->compute_node_histogram(y, sample_indices, 0, num_samples);
    std::vector<std::vector<double>> node_weighted_histogram = gini->get_node_weighted_histogram();

    std::vector<std::vector<double>> expect = {{3.0, 3.0, 3.0}};
    EXPECT_THAT(node_weighted_histogram, ::testing::ContainerEq(expect));
}  

TEST_F(GiniCriterionTest, ComputeNodeImpurityTest) {
    std::vector<long> y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    unsigned long num_samples = y.size();
    std::vector<unsigned long> sample_indices(num_samples);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);

    gini->compute_node_histogram(y, sample_indices, 0, num_samples);
    gini->compute_node_impurity();
    double impurity = gini->get_node_impurity();

    EXPECT_PRED_FORMAT2(DoubleLE, impurity, double(2.0/3.0));
}

TEST_F(GiniCriterionTest, InitChildrenHistogramTest) {
    std::vector<long> y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    unsigned long num_samples = y.size();
    std::vector<unsigned long> sample_indices(num_samples);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);

    gini->compute_node_histogram(y, sample_indices, 0, num_samples);
    gini->init_children_histogram();
    
    std::vector<std::vector<HistogramType>> left_weighted_histogram = gini->get_left_weighted_histogram();
    std::vector<std::vector<HistogramType>> right_weighted_histogram = gini->get_right_weighted_histogram();

    std::vector<std::vector<HistogramType>> expect1 = {{0.0, 0.0, 0.0}};
    std::vector<std::vector<HistogramType>> expect2 = {{3.0, 3.0, 3.0}};
    EXPECT_THAT(left_weighted_histogram, ::testing::ContainerEq(expect1));
    EXPECT_THAT(right_weighted_histogram, ::testing::ContainerEq(expect2));

    std::vector<HistogramType> left_weighted_num_samples = gini->get_left_weighted_num_samples();
    std::vector<HistogramType> right_weighted_num_samples = gini->get_right_weighted_num_samples();

    std::vector<HistogramType> expect3 = {0.0};
    std::vector<HistogramType> expect4 = {9.0};
    EXPECT_THAT(left_weighted_num_samples, ::testing::ContainerEq(expect3));
    EXPECT_THAT(right_weighted_num_samples, ::testing::ContainerEq(expect4));
}

TEST_F(GiniCriterionTest, UpdateChildrenHistogramTest) {
    std::vector<long> y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    unsigned long num_samples = y.size();
    std::vector<unsigned long> sample_indices(num_samples);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);

    gini->compute_node_histogram(y, sample_indices, 0, num_samples);
    gini->init_children_histogram();
    gini->update_children_histogram(y, sample_indices, 3);

    std::vector<std::vector<HistogramType>> left_weighted_histogram = gini->get_left_weighted_histogram();
    std::vector<std::vector<HistogramType>> right_weighted_histogram = gini->get_right_weighted_histogram();

    std::vector<std::vector<HistogramType>> expect1 = {{3.0, 0.0, 0.0}};
    std::vector<std::vector<HistogramType>> expect2 = {{0.0, 3.0, 3.0}};
    EXPECT_THAT(left_weighted_histogram, ::testing::ContainerEq(expect1));
    EXPECT_THAT(right_weighted_histogram, ::testing::ContainerEq(expect2));

    std::vector<HistogramType> left_weighted_num_samples = gini->get_left_weighted_num_samples();
    std::vector<HistogramType> right_weighted_num_samples = gini->get_right_weighted_num_samples();

    std::vector<HistogramType> expect3 = {3.0};
    std::vector<HistogramType> expect4 = {6.0};
    EXPECT_THAT(left_weighted_num_samples, ::testing::ContainerEq(expect3));
    EXPECT_THAT(right_weighted_num_samples, ::testing::ContainerEq(expect4));
}

TEST_F(GiniCriterionTest, ComputeImpurityImprovementTest) {
    std::vector<long> y = {0, 0, 0, 1, 1, 1, 2, 2, 2};
    unsigned long num_samples = y.size();
    std::vector<unsigned long> sample_indices(num_samples);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);

    gini->compute_node_histogram(y, sample_indices, 0, num_samples);
    gini->init_children_histogram();
    
    double impurity = gini->compute_impurity_improvement();
    EXPECT_PRED_FORMAT2(DoubleLE, impurity, double(2.0/3.0));
}

} //namespace
