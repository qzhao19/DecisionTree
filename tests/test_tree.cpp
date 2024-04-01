#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "../decision_tree/core/tree.hpp"

namespace {

using ::testing::DoubleLE;

auto calculate_num_classes_list = [](const std::vector<std::vector<std::string>>& classes) {
    std::vector<unsigned long> num_classes_list(classes.size(), 0);
    for (std::size_t o=0; o<classes.size(); o++) {
        num_classes_list[o] = classes[o].size();
    }
    return num_classes_list;
};

class TreeTest : public ::testing::Test {
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

        tree_ = new decisiontree::Tree(num_outputs, num_features, num_classes_list);

    }
    virtual void TearDown() {
        if (tree_) {
            delete tree_;
            tree_ = nullptr;
        }
    }

    decisiontree::Tree* tree_;
};

TEST_F(TreeTest, AddNodeTest) {
    FeatureIndexType feature_index = 0;
    int has_missing_value = -1; 
    FeatureType threshold = 0.0;
    double impurity = 0.666667; 
    double improvement = 0.0; 
    const std::vector<std::vector<HistogramType>>& histogram = {{3.0, 3.0, 3.0}};

    unsigned long node_index = tree_->add_node(false, 0, 0, 
                                                feature_index, 
                                                has_missing_value, 
                                                threshold, 
                                                impurity, 
                                                improvement, 
                                                histogram);
};

} // namespace