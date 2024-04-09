#include <iostream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "../decision_tree/utility/sort.hpp"

namespace {

TEST(SortTest, Sort2VectorsByFirstVector){
    std::vector<double> x = {5.2, 3.3, 1.2, 0.3, 4.8, 3.1, 1.6, 0.2, 4.75};
    std::vector<std::size_t> y = {0, 0, 0, 1, 1, 1, 2, 2, 2};

    decisiontree::sort<double, std::size_t>(x, y, 0, 5);

    std::vector<double> expect1 = {0.3, 1.2, 3.3, 4.8, 5.2, 3.1, 1.6, 0.2, 4.75};
    std::vector<std::size_t> expect2 = {1, 0, 0, 1, 0, 1, 2, 2, 2};

    EXPECT_THAT(x, ::testing::ContainerEq(expect1));
    EXPECT_THAT(y, ::testing::ContainerEq(expect2));
}

}
