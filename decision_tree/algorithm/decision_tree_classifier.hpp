#ifndef ALGORITHM_DECISION_TREE_CLASSIFIER_HPP_
#define ALGORITHM_DECISION_TREE_CLASSIFIER_HPP_

#include "common/prereqs.hpp"
#include "core/builder.hpp"
#include "core/splitter.hpp"
#include "core/tree.hpp"
#include "utility/math.hpp"
#include "utility/random.hpp"

namespace decisiontree {

/**
 * @brief 
*/
class DecisionTreeClassifier {
private:
    std::string split_policy_;
    std::string criterion_;

    int random_seed_;
    int max_depth_;
    int max_num_features_;
    int min_samples_split_;
    int min_samples_leaf_;
    double min_weight_fraction_leaf_;
    bool class_balanced_;
    std::vector<std::string> feature_names_;
    std::vector<std::vector<std::string>> class_labels_;
    std::shared_ptr<std::vector<double>> class_weight_ptr_;

    NumFeaturesType num_features_;
    NumOutputsType num_outputs_;
    NumClassesType max_num_classes_;
    std::vector<NumClassesType> num_classes_list_;

    decisiontree::RandomState random_state_;
    decisiontree::Splitter splitter_;
    decisiontree::Tree tree_;
    std::shared_ptr<decisiontree::DepthFirstTreeBuilder> builder_;

public:
    DecisionTreeClassifier(std::vector<std::string> feature_names,
                           std::vector<std::vector<std::string>> class_labels,
                           int random_seed = 0,
                           int max_depth = 4, 
                           int max_num_features = -1, 
                           int min_samples_split = 2, 
                           int min_samples_leaf = 1,
                           double min_weight_fraction_leaf = 0.0, 
                           bool class_balanced = true,
                           std::string criterion = "gini", 
                           std::string split_policy = "best",
                           std::shared_ptr<std::vector<double>> class_weight_ptr = nullptr):
                        feature_names_(feature_names),
                        class_labels_(class_labels),
                        max_depth_(max_depth),
                        max_num_features_(max_num_features),
                        min_samples_split_(min_samples_split),
                        min_samples_leaf_(min_samples_leaf),
                        min_weight_fraction_leaf_(min_weight_fraction_leaf),
                        random_seed_(random_seed),
                        class_balanced_(class_balanced),
                        criterion_(criterion),
                        split_policy_(split_policy),
                        class_weight_ptr_(class_weight_ptr), 
                        num_features_(feature_names.size()), 
                        num_outputs_(class_labels.size()) {
        
        num_classes_list_.resize(class_labels_.size(), 0);
        for (std::size_t o=0; o < class_labels_.size(); o++) {
            num_classes_list_[o] = class_labels_[o].size();
        }

        max_num_classes_ = *std::max_element(begin(num_classes_list_),
                                             end(num_classes_list_));
    };
    
    ~DecisionTreeClassifier() {};

    void fit(const std::vector<FeatureType>& X, 
             const std::vector<ClassType>& y) {
        NumSamplesType num_samples = y.size() / num_outputs_;

        // check max_depth
        if (max_depth_ < 0) {
            throw std::invalid_argument("max_depth must be positive");
        }
        NumSamplesType max_depth = static_cast<NumSamplesType>(max_depth_);

        // check min_samples_leaf
        if (min_samples_leaf_ < 0) {
            throw std::invalid_argument("min_samples_leaf must be positive");
        }
        NumSamplesType min_samples_leaf = static_cast<NumSamplesType>(min_samples_leaf_);

        // check min_samples_split
        if (min_samples_split_ < 0) {
            throw std::invalid_argument("min_samples_split must be positive");
        }
        NumSamplesType min_samples_split = static_cast<NumSamplesType>(min_samples_split_);
        min_samples_split_ = std::max(min_samples_split_, 2 * min_samples_leaf_);

        // check max_num_features
        NumFeaturesType max_num_features;
        if (max_num_features_ == -1) {
            max_num_features = num_features_;
        }
        else if (max_num_features_ > 0) {
            max_num_features = static_cast<NumFeaturesType>(max_num_features_);
        }

        // check class_weight
        std::vector<double> class_weight(num_outputs_ * max_num_classes_, 1.0);;
        if (class_balanced_) {
            for (unsigned long o = 0; o < num_outputs_; ++o) { // process each output independently
                std::vector<long> bincount(num_classes_list_[o], 0);
                for (unsigned long i = 0; i < num_samples; ++i) {
                    bincount[y[i * num_outputs_ + o]]++;
                }
                for (unsigned long c = 0; c < num_classes_list_[o]; ++c) {
                    class_weight[o * max_num_classes_ + c] =
                        (static_cast<double>(num_samples) / bincount[c]) / num_classes_list_[o];
                }
            }
        }
        else {
            if (class_weight_ptr_ == nullptr) {
                throw std::invalid_argument(
                    "If 'class_balanced' is false, must provide a smart pointer to a class weight."
                    "Weights associated with classes in the form {weight, weight, ..., }."
                );
            }
            class_weight = *class_weight_ptr_;
        }

        // check min_weight_leaf
        NumSamplesType min_weight_leaf;
        if (class_balanced_) {
            min_weight_leaf = static_cast<NumSamplesType>(min_weight_fraction_leaf_ * num_samples);
        } 
        else {
            double sum_weight = std::accumulate(class_weight.begin(), class_weight.end(), 0.0);
            min_weight_leaf = static_cast<NumSamplesType>(min_weight_fraction_leaf_ * sum_weight);
        }

        // init criterion
        if (CRITERIA_CLF.find(criterion_) == CRITERIA_CLF.end()) {
            throw std::invalid_argument("Criterion must be either 'gini' or 'entropy'.");
        }

        if (random_seed_ == -1) {
            random_state_ = decisiontree::RandomState();
        }
        else {
            random_state_ = decisiontree::RandomState(random_seed_);
        }

        splitter_ = decisiontree::Splitter(num_outputs_,  
                                           num_samples,
                                           num_features_,
                                           max_num_features,
                                           max_num_classes_,
                                           class_weight,
                                           num_classes_list_,
                                           criterion_,
                                           split_policy_,
                                           random_state_);
        
        tree_ = decisiontree::Tree(num_outputs_, 
                                   num_features_, 
                                   num_classes_list_);

        builder_ = std::make_shared<decisiontree::DepthFirstTreeBuilder>(max_depth, 
                                                                        min_samples_split, 
                                                                        min_samples_leaf, 
                                                                        min_weight_leaf,
                                                                        class_weight,
                                                                        splitter_, 
                                                                        tree_);
        builder_->build(X, y, num_samples);
    };

    const std::vector<double> predict_proba(const std::vector<FeatureType>& X) {
        NumSamplesType num_samples = X.size() / num_features_;
        std::vector<double> proba;
        builder_->tree_.predict_proba(X, num_samples, proba);

        return proba;
    };

    const std::vector<ClassType> predict(const std::vector<FeatureType>& X) {
        NumSamplesType num_samples = X.size() / num_features_;
        std::vector<double> proba;
        builder_->tree_.predict_proba(X, num_samples, proba);

        std::vector<ClassType> label(num_samples * num_outputs_, 0);
        for (IndexType i = 0; i < num_samples; ++i) {
            for (IndexType o = 0; o < num_outputs_; ++o) {
                label[i*num_outputs_ + o] = decisiontree::argmax<FeatureType, ClassType>(
                    &proba[i*num_outputs_*max_num_classes_ + o*max_num_classes_], num_classes_list_[o]
                );
            }
        }

        return label;
    };

    const std::vector<double> compute_feature_importance() {
        std::vector<double> f_importances;
        builder_->tree_.compute_feature_importance(f_importances);
        return f_importances;
    };

    void print_node_info() {
        builder_->tree_.print_node_info();
    };  

};

} // namespace

#endif // ALGORITHM_DECISION_TREE_CLASSIFIER_HPP_