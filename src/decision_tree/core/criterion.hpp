#ifndef CORE_CRITERION_HPP
#define CORE_CRITERION_HPP

#include "prereqs.hpp"

namespace decisiontree {

/**
 * @brief 
*/ 
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

protected:
    /**
     * impurity of a weighted class histogram
     * The Gini Index is then defined as:
     *  - index = 1 - sum_{k=0}^{k-1} count_k ** 2, where 
     * @param histogram sum of the weighted count of each label
     * 
    */
    double compute_impurity(const std::vector<HistogramType>& histogram) {
        double cnt;
        double sum_cnt = 0.0;
        double sum_cnt_squared = 0.0;

        for (IndexType c = 0; c < histogram.size(); ++c) {
            cnt = static_cast<double>(histogram[c]);
            sum_cnt += cnt;
            sum_cnt_squared += cnt * cnt;
        }
        return (sum_cnt > 0.0) ? (1.0 - sum_cnt_squared / (sum_cnt*sum_cnt)) : 0.0;
    };

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

    /**
     * compute weighted class histograms for current node.
     * 
     * @param y target stored as a buffer
     * @param sample_indices  mask on the samples. 
     *      Indices of the samples in X and y we want to use, 
     *      where sample_indices[start:end] correspond to the 
     *      samples in this node
     * @param start the first sample to use in the mask
     * @param end the last sample to use in the mask
    */
    void compute_node_histogram(const std::vector<ClassType>& y, 
                                const std::vector<SampleIndexType>& sample_indices, 
                                SampleIndexType start, 
                                SampleIndexType end) {
        // for each output
        for (IndexType o = 0; o < num_outputs_; o++) {
            // 1d array to hold the class histogram
            // count labels for each output
            std::vector<HistogramType> histogram(max_num_classes_, 0);
            for (IndexType i = start; i < end; i++) {
                histogram[y[sample_indices[i] * num_outputs_ + o]]++;
            }

            // class_weight_ is set to be 1.0
            HistogramType weighted_cnt;
            node_weighted_num_samples_[o] = 0.0;
            for (NumClassesType c = 0; c < num_classes_list_[o]; c++) {
                weighted_cnt = class_weight_[o * max_num_classes_ + c] * histogram[c];
                node_weighted_histogram_[o][c] = weighted_cnt;
                node_weighted_num_samples_[o] += weighted_cnt;
            }
        }
    }


};

}

#endif //CORE_CRITERION_HPP