#ifndef CORE_SPLITTER_HPP_
#define CORE_SPLITTER_HPP_

#include "../common/prereqs.hpp"
#include "../utility/random.hpp"
#include "../utility/sort.hpp"

#include "criterion.hpp"

namespace decisiontree {

/**
 * @brief
*/
class Splitter {
private:
    NumFeaturesType num_features_;
    NumSamplesType num_samples_;
    NumFeaturesType max_num_features_;
    std::string split_policy_;
    RandomState random_state_;

    SampleIndexType start_;
    SampleIndexType end_;
    std::vector<SampleIndexType> sample_indices_;

protected:
    void random_split_feature(const std::vector<FeatureType>& X, 
                              const std::vector<ClassType>& y,
                              std::vector<SampleIndexType>& sample_indices, 
                              FeatureIndexType feature_index, 
                              SampleIndexType& partition_index,
                              FeatureType& partition_threshold,
                              double& improvement, 
                              int& has_missing_value) {
        
        // copy f_X = X[sample_indices[start:end], feature_index] 
        // as the selected training data for the current node
        NumSamplesType num_samples = end_ - start_;
        std::vector<FeatureType> f_X(num_samples);
        for (IndexType i = 0; i < num_samples; ++i) {
            f_X[i] = X[sample_indices[i]*num_features_ + feature_index];
        }
        
        // check the missing value and shift missing value and its index to the left
        SampleIndexType missing_value_index = 0;
        for (IndexType i = 0; i < num_samples; ++i) {
            if (std::isnan(f_X[i])) {
                std::swap(f_X[i], f_X[missing_value_index]);
                std::swap(sample_indices[i], sample_indices[missing_value_index]);
                missing_value_index++;
            }
        }

        // if all samples have missing value, cannot split feature
        if (missing_value_index == num_samples) {
            return ;
        }

        // check constant feature in the range of [missing_value_index:num_samples]
        // we start to check constant feature from non-missing value
        FeatureType fx_min, fx_max;
        fx_min = fx_max = f_X[missing_value_index];
        for (IndexType i = missing_value_index + 1; i < num_samples; ++i) {
            if (f_X[i] > fx_max) {
                fx_max = f_X[i];
            }
            else if (f_X[i] < fx_min) {
                fx_min = f_X[i];
            }
        }

        // ---Split just based on missing values---
        if ((missing_value_index > 0) && 
            ((fx_min + EPSILON > fx_max) || 
             (random_state_.uniform_int(0, num_samples) < (missing_value_index-1)))) {
            throw std::runtime_error("Not Implemented");
        }
        // ---Split based on threshold---
        else {
            // not constant feature
            if (fx_min + EPSILON < fx_max) {
                // random a threshold
                // excludes fx_min, fx_max, with uniform_real(low, high), low is inclusive and high is exclusive
                partition_threshold = random_state_.uniform_real(fx_min + EPSILON, fx_max);

                // partition sample indices such that f_X[indices[np - 1]] <= threshold < f_X[indices[np]]
                SampleIndexType index = missing_value_index, next_index = num_samples;
                while (index < next_index) {
                    if (f_X[index] <= partition_threshold) {
                        ++index;
                    }
                    else {
                        --next_index;
                        std::swap(f_X[index], f_X[next_index]);
                        std::swap(sample_indices[index], sample_indices[next_index]);
                    }
                }

                // no missing value
                if (missing_value_index == 0) {
                    criterion_.init_children_histogram();
                    criterion_.update_children_histogram(y, sample_indices, next_index);
                    criterion_.compute_children_impurity();
                    double impurity_improvement = criterion_.compute_impurity_improvement();

                    partition_index = start_ + next_index;
                    improvement = impurity_improvement;
                    has_missing_value = -1;
                }
                else if (missing_value_index > 0) {
                    throw std::runtime_error("Not Implemented");
                }
            }
        }
    };

    void best_split_feature(const std::vector<FeatureType>& X, 
                            const std::vector<ClassType>& y, 
                            std::vector<SampleIndexType>& sample_indices, 
                            FeatureIndexType feature_index, 
                            SampleIndexType& partition_index,
                            FeatureType& partition_threshold,
                            double& improvement, 
                            int& has_missing_value) {
        
        // copy f_X = X[sample_indices[start:end], feature_index] 
        // as the selected training data for the current node
        NumSamplesType num_samples = end_ - start_;
        std::vector<FeatureType> f_X(num_samples);
        for (IndexType i = 0; i < num_samples; ++i) {
            f_X[i] = X[sample_indices[i]*num_features_ + feature_index];
        }
        
        // check the missing value and shift missing value and its index to the left
        SampleIndexType missing_value_index = 0;
        for (IndexType i = 0; i < num_samples; ++i) {
            if (std::isnan(f_X[i])) {
                std::swap(f_X[i], f_X[missing_value_index]);
                std::swap(sample_indices[i], sample_indices[missing_value_index]);
                missing_value_index++;
            }
        }

        // if all samples have missing value, cannot split feature
        if (missing_value_index == num_samples) {
            return ;
        }

        // if samples have missing value
        if (missing_value_index > 0) {
            // compute class histogram, impurity and improvement for samples with missing values
            criterion_.compute_node_histogram_missing(y, sample_indices, missing_value_index);
            criterion_.compute_node_impurity_missing();
            improvement = criterion_.compute_impurity_improvement_missing();
            // pass all samples with missing values to the left child
            // pass all samples with non-missing values to the right child
            has_missing_value = 0;
            partition_threshold = std::numeric_limits<FeatureType>::quiet_NaN();
            partition_index = start_ + missing_value_index;

            if (criterion_.get_node_impurity_non_missing() < EPSILON) {
                return;
            }

        }

        // ---Split based on threshold---
        // check constant feature in the range of [missing_value_index:num_samples]
        // we start to check constant feature from non-missing value
        FeatureType fx_min, fx_max;
        fx_min = fx_max = f_X[missing_value_index];
        for (IndexType i = missing_value_index + 1; i < num_samples; ++i) {
            if (f_X[i] > fx_max) {
                fx_max = f_X[i];
            }
            else if (f_X[i] < fx_min) {
                fx_min = f_X[i];
            }
        }

        // not constant feature
        if (fx_min + EPSILON < fx_max) {

            if (missing_value_index == 0) {
                criterion_.init_children_histogram();
            }
            else if (missing_value_index > 0) {
                criterion_.init_children_histogram_non_missing();
            }

            // sort f_X and corresponding sample_indices by soring f_X
            // missing values are at the beginnig of sample_indices
            sort<FeatureType, FeatureIndexType>(f_X, sample_indices, missing_value_index, num_samples); 

            // find threshold
            // init index and next_index for the position of last and next potential split position
            SampleIndexType index = missing_value_index;
            SampleIndexType next_index = missing_value_index;

            double max_improvement = 0.0;
            FeatureType max_partition_threshold = 0.0;
            SampleIndexType max_partition_index = missing_value_index;
            while (next_index < num_samples) {
                // if remaining f_X are constant, stop
                // f_X[missing_value_indice, num_samples] is sorted
                if (f_X[next_index] + EPSILON >= f_X[num_samples - 1]) {
                    break;
                }

                // skip constant f_X value, f_X is already sorted
                while ((next_index + 1 < num_samples) && (f_X[next_index] + EPSILON >= f_X[next_index + 1])) {
                    next_index++;
                }
                // go to the next position
                next_index++;

                // update class histograms from current indice to the new indice (correspond to threshold)
                criterion_.update_children_histogram(y, sample_indices, next_index);

                // compute impurity for left child and right child
                criterion_.compute_children_impurity();
                
                // compute impurity improvement
                double impurity_improvement = 0.0;
                if (missing_value_index == 0) {
                    impurity_improvement = criterion_.compute_impurity_improvement();
                    // std::cout << "impurity_improvement = " << impurity_improvement << std::endl;
                }
                else if (missing_value_index > 0) {
                    impurity_improvement = criterion_.compute_impurity_improvement_non_missing();
                }

                if (impurity_improvement > max_improvement) {
                    max_improvement = impurity_improvement;
                    max_partition_threshold = (f_X[index] + f_X[next_index]) / 2.0;
                    max_partition_index = start_ + next_index; 
                    // std::cout << "current position = " << index << ", value = " << f_X[index] << 
                    //              "next position = " << next_index << ", value = " << f_X[next_index] << std::endl;
                }
                
                // if right node impurity is 0.0 stop
                if (criterion_.get_right_impurity() < EPSILON) {
                    break;
                }
                index = next_index;
            }

            // samples without missing values
            if (missing_value_index == 0) {
                partition_index = max_partition_index;
                partition_threshold = max_partition_threshold;
                improvement = max_improvement;
                has_missing_value = -1;
            }
            else if (missing_value_index > 0) {
                // call compute_children_impurity_missing 
                criterion_.compute_children_impurity_missing();

                // compute left and right improvement for samples with missing values
                double left_impurity_improvement = criterion_.compute_left_impurity_improvement_missing();
                double right_impurity_improvement = criterion_.compute_right_impurity_improvement_missing();

                if (left_impurity_improvement > right_impurity_improvement) {
                    // add missing values to left child
                    if (improvement < left_impurity_improvement) {
                        improvement = left_impurity_improvement;
                        partition_index = max_partition_index;
                        partition_threshold = max_partition_threshold;
                        has_missing_value = 0;
                    }
                }
                else {
                    // add missing values to left child
                    if (improvement < right_impurity_improvement) {
                        improvement = right_impurity_improvement;
                        has_missing_value = 1;
                        partition_threshold = max_partition_threshold;

                        // move samples with missing values to the end of the sample vector
                        std::vector<SampleIndexType> sample_indice_missing(&sample_indices[0], 
                                                                           &sample_indices[missing_value_index]);
                        std::copy(&sample_indices[missing_value_index], 
                                  &sample_indices[num_samples], 
                                  &sample_indices[0]);
                        std::copy(&sample_indice_missing[0], 
                                  &sample_indice_missing[missing_value_index], 
                                  &sample_indices[num_samples - missing_value_index]);
                        partition_index = max_partition_index - missing_value_index;
                    }
                }
            }
        }
    };

public:
    Criterion criterion_;

public:
    Splitter() {};
    Splitter(NumFeaturesType num_features, 
             NumSamplesType num_samples, 
             NumFeaturesType max_num_features, 
             std::string split_policy, 
             const Criterion& criterion, 
             const RandomState& random_state): num_features_(num_features),
        num_samples_(num_samples), 
        max_num_features_(max_num_features), 
        split_policy_(split_policy),
        criterion_(criterion),
        random_state_(random_state), 
        // init sample index array 
        sample_indices_(num_samples),
        start_(0), 
        end_(num_samples) {
            // init the mask on the samples.
            std::iota(sample_indices_.begin(), sample_indices_.end(), 0);
        };
    ~Splitter() {};

    /**
     * initialize node and compute weighted histograms and impurity for the node.
    */
    void init_node(const std::vector<ClassType>& y, 
                   SampleIndexType start, 
                   SampleIndexType end) {
        
        start_ = start;
        end_ = end;
        criterion_.compute_node_histogram(y, sample_indices_, start, end);
        criterion_.compute_node_impurity();
    }


    void split_node(const std::vector<FeatureType>& X, 
                    const std::vector<ClassType>& y, 
                    FeatureIndexType& feature_index, 
                    SampleIndexType& partition_index,
                    FeatureType& partition_threshold,
                    double& improvement, 
                    int& has_missing_value) {

        // copy current sample_indices = sample_indices[start_, end_]
        // lookup-table to the training data X, y
        NumSamplesType num_samples = end_ - start_;
        std::vector<SampleIndexType> f_sample_indice(num_samples);
        std::copy(&sample_indices_[start_], &sample_indices_[end_], f_sample_indice.begin());

        // loop: k random features (k defined by max_num_features)
        // loop all features in random order

        // sampling features without replacement in an iterative way
        // init a feature index array
        std::vector<FeatureIndexType> f_indices(num_features_);
        std::iota(f_indices.begin(), f_indices.end(), 0);

        // i is in range of [0, num_features - 1]
        FeatureIndexType i = num_features_;
        while ((i > (num_features_ - max_num_features_)) || 
               (improvement < EPSILON && i > 0)) {
            
            // std::cout << "current index i = " << i << std::endl;
            FeatureIndexType j = 0;
            if (i > 0) {
                j = static_cast<FeatureIndexType>(random_state_.uniform_int(0, i));
            }
            // std::cout << "random index j = " << j << std::endl;
            --i;
            std::swap(f_indices[i], f_indices[j]);
            FeatureIndexType f_index = f_indices[i];

            // split feature 
            int f_has_missing_value = 0;
            double f_improvement = improvement;
            SampleIndexType f_partition_index = 0;
            FeatureType f_partition_threshold = 0.0;

            if (split_policy_ == "best") {
                best_split_feature(X, y, 
                                   f_sample_indice, 
                                   f_index, 
                                   f_partition_index, 
                                   f_partition_threshold, 
                                   f_improvement, 
                                   f_has_missing_value);
            }
            else if (split_policy_ == "random") {
                random_split_feature(X, y, 
                                     f_sample_indice, 
                                     f_index,
                                     f_partition_index,
                                     f_partition_threshold,
                                     f_improvement,
                                     f_has_missing_value);
            }
            else {
                throw std::invalid_argument(
                    "Supported strategies are 'best' to choose the best "
                    "split and 'random' to choose the best random split.");
            }

            if (f_improvement > improvement) {
                feature_index = f_index;
                partition_index = f_partition_index;
                partition_threshold = f_partition_threshold;
                improvement = f_improvement;
                has_missing_value = f_has_missing_value;
                // replace sample_indices_ with ordered f_sample_indices
                std::copy(f_sample_indice.begin(), f_sample_indice.end(), &sample_indices_[start_]);
            }
        }
    }

};

} //namespace

#endif // CORE_SPLITTER_HPP_