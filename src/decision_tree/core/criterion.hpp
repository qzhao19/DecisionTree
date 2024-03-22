#ifndef CORE_CRITERION_HPP
#define CORE_CRITERION_HPP

#include "prereqs.hpp"

namespace decisiontree {

class Gini {
private:
    NumOutputsType num_outputs_;
    NumSamplesType num_samples_;
    NumClassesType max_num_classes_;
    std::vector<NumClassesType> num_classes_list_;
    std::vector<ClassWeightType> class_weight_;

    // weighted histogram in the parent node
    std::vector<std::vector<HistogramType>> node_weighted_histogram_;
    // weighted histogram in left node with values smaller than threshold
    std::vector<std::vector<HistogramType>> left_weighted_histogram_;
    // weighted histogram in right node with values bigger than threshold
    std::vector<std::vector<HistogramType>> right_weighted_histogram_;

    // impurity of in the current node
    std::vector<double> node_impurity_;
    // impurity of in the left node with values smaller than threshold
    std::vector<double> left_impurity_;
    // impurity of in the right node with values bigger that threshold
    std::vector<double> right_impurity_;

    // weighted number of samples in the parent node, left child and right child
    std::vector<HistogramType> node_weighted_num_samples_;
    std::vector<HistogramType> left_weighted_num_samples_;
    std::vector<HistogramType> right_weighted_num_samples_;


public:
    Gini(NumOutputsType num_outputs, 
         NumSamplesType num_samples, 
         NumClassesType max_num_classes, 
         std::vector<NumClassesType> num_classes_list, 
         std::vector<ClassWeightType> class_weight): num_outputs_(num_outputs), 
            num_samples_(num_samples), 
            max_num_classes_(max_num_classes),
            num_classes_list_(num_classes_list),
            class_weight_(class_weight), 

            // create and initialize histograms
            node_weighted_histogram_(num_outputs, std::vector<HistogramType>(max_num_classes_, 0.0)), 
            node_weighted_num_samples_(num_outputs, 0.0),
            node_impurity_(num_outputs, 0.0), 

            left_weighted_histogram_(num_outputs, std::vector<HistogramType>(max_num_classes_, 0.0)),
            left_weighted_num_samples_(num_outputs, 0.0),
            left_impurity_(num_outputs, 0.0), 

            right_weighted_histogram_(num_outputs, std::vector<HistogramType>(max_num_classes_, 0.0)), 
            right_weighted_num_samples_(num_outputs, 0.0), 
            right_impurity_(num_outputs, 0.0) {};

    ~Gini() {};


    void compute_node_histogram(const std::vector<ClassType>& y, 
                                const std::vector<SampleIndexType>& sample_indices, 
                                SampleIndexType start, 
                                SampleIndexType end) {
        
        





    }



};

}

#endif //CORE_CRITERION_HPP