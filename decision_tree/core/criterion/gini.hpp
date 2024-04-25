#ifndef CORE_CRITERION_GINI_HPP_
#define CORE_CRITERION_GINI_HPP_

#include "../../common/prereqs.hpp"
#include "base.hpp"

namespace decisiontree {

class Gini : public Criterion {
private:
    NumOutputsType num_outputs_;
    NumSamplesType num_samples_;
    NumClassesType max_num_classes_;
    std::vector<NumClassesType> num_classes_list_;
    std::vector<ClassWeightType> class_weight_;

protected:
    /**
     * @brief override method to compute impurity of a weighted class histogram
     * The Gini Index is then defined as:
     *  - index = 1 - sum_{k=0}^{k-1} count_k ** 2, where 
     * @param histogram sum of the weighted count of each label
    */
    double compute_impurity(const std::vector<HistogramType>& histogram) override {
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
    Gini() {};
    Gini(NumOutputsType num_outputs, 
                  NumSamplesType num_samples, 
                  NumClassesType max_num_classes, 
                  std::vector<NumClassesType> num_classes_list, 
                  std::vector<ClassWeightType> class_weight): Criterion(num_outputs, 
                    num_samples, 
                    max_num_classes, 
                    num_classes_list, 
                    class_weight) {};
    ~Gini() {};

};

}
#endif // CORE_CRITERION_GINI_HPP_