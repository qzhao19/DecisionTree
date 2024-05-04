#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "decision_tree/utility/math.hpp"

namespace {

TEST(SortTest, Sort2VectorsByFirstVector){
    std::vector<double> x = {5.2, 3.3, 1.2, 0.3, 4.8, 3.1, 1.6, 0.2, 4.75};
    long max_index;
    max_index = decisiontree::argmax<double, long>(&x[0], 9);
    EXPECT_THAT(max_index, 0);
};

}