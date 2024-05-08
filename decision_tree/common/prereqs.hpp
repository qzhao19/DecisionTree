#ifndef COMMON_PREREQS_HPP_
#define COMMON_PREREQS_HPP_

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stack>
#include <string>
#include <stdexcept>
#include <utility>
#include <unordered_set>
#include <vector>

using FeatureType = double;
using ClassType = long;

using ClassWeightType = double;
using HistogramType = double;

using NumSamplesType = unsigned long;
using NumFeaturesType = unsigned long;
using NumOutputsType = unsigned long;
using NumClassesType = unsigned long;

using IndexType = unsigned long;
using NodeIndexType = unsigned long;
using FeatureIndexType = unsigned long;
using SampleIndexType = unsigned long;

using TreeDepthType = unsigned long;

const double EPSILON = 1e-7;

const std::unordered_set<std::string> CRITERIA_CLF = {"gini", "entropy"};

#endif // COMMON_PREREQS_HPP_