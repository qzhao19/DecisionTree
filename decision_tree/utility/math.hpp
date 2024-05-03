#ifndef UTILITY_MATH_HPP_
#define UTILITY_MATH_HPP_

#include "common/prereqs.hpp"

namespace decisiontree {

template<typename FeatureType, typename ClassType>
ClassType argmax(FeatureType* x, long size) {
    ClassType max_index = 0;
    FeatureType max_value = x[max_index];

    for (unsigned long i = 0; i < size; i++) {
        if (x[i] > max_value) {
            max_index = i;
            max_value = x[max_index];
        }
    }
    return max_index;
};

}
#endif // UTILITY_MATH_HPP_