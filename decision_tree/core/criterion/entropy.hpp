#ifndef CORE_CRITERION_ENTROPY_HPP_
#define CORE_CRITERION_ENTROPY_HPP_

#include "common/prereqs.hpp"
#include "base.hpp"

namespace decisiontree {

class Entropy : public Criterion {
protected:
    /**
     * @brief override method to compute impurity of a weighted class histogram
     * The Entropy is then defined as:
     *  - count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k) 
     *  - cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
     *  
     * @param histogram sum of the weighted count of each label
    */
    double compute_impurity(const std::vector<HistogramType>& histogram) override {
        double cnt;
        double sum_cnt = 0.0;
        double entropy = 0.0;
        for (IndexType c = 0; c < histogram.size(); ++c) {
            cnt = static_cast<double>(histogram[c]);
            sum_cnt += cnt;
            if (cnt > 0.0) {
                cnt /= sum_cnt;
                entropy -= cnt * std::log2(cnt);
            }
        }
        return entropy;
    };

public:
    Entropy() {};
    Entropy(NumOutputsType num_outputs, 
            NumSamplesType num_samples, 
            NumClassesType max_num_classes, 
            std::vector<NumClassesType> num_classes_list, 
            std::vector<ClassWeightType> class_weight): Criterion(num_outputs, 
            num_samples, 
            max_num_classes, 
            num_classes_list, 
            class_weight) {};
    ~Entropy() {};

};

}
#endif // CORE_CRITERION_ENTROPY_HPP_