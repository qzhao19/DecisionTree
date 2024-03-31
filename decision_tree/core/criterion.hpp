#ifndef CORE_CRITERION_HPP_
#define CORE_CRITERION_HPP_

#include "../common/prereqs.hpp"

namespace decisiontree {

/**
 * @brief 
*/ 
class Criterion {
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

    SampleIndexType threshold_index_;

protected:
    /**
     * @brief impurity of a weighted class histogram
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
    Criterion(NumOutputsType num_outputs, 
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
            right_impurity_(num_outputs, 0.0), 
            threshold_index_(0) {};

    ~Criterion() {};

    /**
     * @brief weighted class histograms for current node.
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

    /**
     * @brief Evaluate the impurity of the current node.
    */
    void compute_node_impurity() {
        // for each output
        for (IndexType o = 0; o < num_outputs_; o++) {
            node_impurity_[o] = compute_impurity(node_weighted_histogram_[o]);
        }
    } 

    /**
     * @brief compute impurity for all outputs of samples for 
     *  right child and right child
    */
    void compute_children_impurity() {
        // for each output
        for (IndexType o = 0; o < num_outputs_; o++) {
            left_impurity_[o] = compute_impurity(left_weighted_histogram_[o]);
            right_impurity_[o] = compute_impurity(right_weighted_histogram_[o]);
        }
    }

    /**
     * @brief initialize class histograms for all outputs 
     *  for using a threshold on samples with values,
    */
    void init_children_histogram() {
        // for each output
        for (IndexType o = 0; o < num_outputs_; o++) {
            // init class histogram for left child and right child value of 
            // left child is 0, value of right child is current node value
            for (NumClassesType c = 0; c < num_classes_list_[o]; c++) {
                left_weighted_histogram_[o][c] = 0.0;
                right_weighted_histogram_[o][c] = node_weighted_histogram_[o][c];
            }

            left_weighted_num_samples_[o] = 0.0;
            right_weighted_num_samples_[o] = node_weighted_num_samples_[o];
        }
        // update current position
        threshold_index_ = 0;
    }

    /**
     * @brief update class histograms of child nodes with new threshold
    */
    void update_children_histogram(const std::vector<ClassType>& y, 
                                   const std::vector<SampleIndexType>& sample_indices,
                                   SampleIndexType new_threshold_index) {
        // for each output
        for (IndexType o = 0; o < num_outputs_; o++) {
            std::vector<HistogramType> histogram(max_num_classes_, 0);

            for (IndexType i = threshold_index_; i < new_threshold_index; i++) {
                histogram[y[sample_indices[i] * num_outputs_ + o]]++;
            }

            // add histogram for samples[index:new_index] to class histogram
            // for samples[0:index] with value < threshold ==> left child
            // subtract histogram for samples[index:new_index] to class histogram
            // for samples[0:index] with value >= threshold ==> right child
            HistogramType weighted_cnt;
            for (NumClassesType c = 0; c < num_classes_list_[o]; c++) {
                weighted_cnt = class_weight_[o * max_num_classes_ + c] * histogram[c];
                // left child
                left_weighted_histogram_[o][c] += weighted_cnt;
                left_weighted_num_samples_[o] += weighted_cnt;

                // right child
                right_weighted_histogram_[o][c] -= weighted_cnt;
                right_weighted_num_samples_[o] -= weighted_cnt;
            }
        }
        // update current threshold index
        threshold_index_ = new_threshold_index;
    }

    /**
     * This method computes the improvement in impurity when a split occurs.
     * The weighted impurity improvement equation is the following:
     *          N_t / N * (impurity - N_t_R / N_t * right_impurity
     *                              - N_t_L / N_t * left_impurity)
     * where N is the total number of samples, N_t is the number of samples
     * at the current node, N_t_L is the number of samples in the left child,
     * and N_t_R is the number of samples in the right child
    */
    double compute_impurity_improvement() {
        std::vector<double> impurity_improvement(num_outputs_, 0.0);
        for (IndexType o = 0; o < num_outputs_; ++o) {
            impurity_improvement[o] += (node_weighted_num_samples_[o] / num_samples_) * (node_impurity_[o] - 
                left_weighted_num_samples_[o] / node_weighted_num_samples_[o] * left_impurity_[o] - 
                    right_weighted_num_samples_[o] / node_weighted_num_samples_[o] * right_impurity_[o]);
        }

        return std::accumulate(impurity_improvement.begin(), impurity_improvement.end(), 0.0) / impurity_improvement.size();
    }

    /**
     * @brief interface method to return weighted histogram of the current node
    */
    const std::vector<std::vector<HistogramType>> get_node_weighted_histogram() {
        return node_weighted_histogram_;
    }

    const std::vector<std::vector<HistogramType>> get_left_weighted_histogram() {
        return left_weighted_histogram_;
    }

    const std::vector<std::vector<HistogramType>> get_right_weighted_histogram() {
        return right_weighted_histogram_;
    }

    const std::vector<HistogramType> get_node_weighted_num_samples() {
        return node_weighted_num_samples_;
    }

    const std::vector<HistogramType> get_left_weighted_num_samples() {
        return left_weighted_num_samples_;
    }

    const std::vector<HistogramType> get_right_weighted_num_samples() {
        return right_weighted_num_samples_;
    }

    const double get_node_impurity() {
        return std::accumulate(node_impurity_.begin(), 
                               node_impurity_.end(), 
                               0.0) / num_outputs_;
    }

    const double get_left_impurity() {
        return std::accumulate(left_impurity_.begin(), 
                               left_impurity_.end(), 
                               0.0) / num_outputs_;
    }

    const double get_right_impurity() {
        return std::accumulate(right_impurity_.begin(), 
                               right_impurity_.end(), 
                               0.0) / num_outputs_;
    }


};

}
#endif //CORE_CRITERION_HPP_