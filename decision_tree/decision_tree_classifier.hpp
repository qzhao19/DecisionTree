#ifndef DECISION_TREE_CLASSIFIER_HPP_
#define DECISION_TREE_CLASSIFIER_HPP_

#include "decision_tree/common/prereqs.hpp"
#include "decision_tree/core/builder.hpp"
#include "decision_tree/core/criterion.hpp"
#include "decision_tree/core/splitter.hpp"
#include "decision_tree/core/tree.hpp"
#include "decision_tree/utils/random.hpp"

namespace {

class DecisionTreeClassifier {
private:
    std::string split_policy_;
    std::string criterion_option_;

    int random_state_seed_;
    int max_depth_;
    int max_num_features_;
    int min_samples_split_;
    int min_samples_leaf_;
    double min_weight_fraction_leaf_;
    bool class_balanced_;
    std::vector<std::string> feature_names_;
    std::vector<std::vector<std::string>> class_labels_;
    std::shared_ptr<std::vector<double>> class_weight_ptr_;

    decisiontree::RandomState random_state_;
    decisiontree::Criterion criterion_;

public:
    DecisionTreeClassifier(std::vector<std::string> feature_names,
                           std::vector<std::vector<std::string>> class_labels,
                           int max_depth = 4, 
                           int max_num_features = -1, 
                           int min_samples_split = 2, 
                           int min_samples_leaf = 1,
                           double min_weight_fraction_leaf = 0.0, 
                           int random_state_seed = -1,
                           bool class_balanced = true,
                           std::string criterion_option = "gini", 
                           std::string split_policy = "best",
                           std::shared_ptr<std::vector<double>> class_weight_ptr = nullptr):
                        feature_names_(feature_names),
                        class_labels_(class_labels),
                        max_depth_(max_depth),
                        max_num_features_(max_num_features),
                        min_samples_split_(min_samples_split),
                        min_samples_leaf_(min_samples_leaf),
                        min_weight_fraction_leaf_(min_weight_fraction_leaf),
                        random_state_seed_(random_state_seed),
                        class_balanced_(class_balanced),
                        criterion_option_(criterion_option),
                        split_policy_(split_policy),
                        class_weight_ptr_(class_weight_ptr) {};
    
    ~DecisionTreeClassifier() {}


    void fit(const std::vector<FeatureType> & X, 
             const std::vector<ClassType>& y) {
        NumFeaturesType num_features = feature_names_.size();
        NumOutputsType num_outputs = class_labels_.size();
        NumSamplesType num_samples = y.size() / num_outputs;

        std::vector<NumClassesType> num_classes_list(class_labels_.size(), 0);
        for (std::size_t o=0; o < class_labels_.size(); o++) {
            num_classes_list[o] = class_labels_[o].size();
        }

        NumClassesType max_num_classes = *std::max_element(begin(num_classes_list),
                                                            end(num_classes_list));
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
            max_num_features = num_features;
        }
        else if (max_num_features_ > 0) {
            max_num_features = static_cast<NumFeaturesType>(max_num_features_);
        }

        // check class_weight
        std::vector<double> class_weight(num_outputs * max_num_classes, 1.0);;
        if (class_balanced_) {
            for (unsigned long o = 0; o < num_outputs; ++o) { // process each output independently
                std::vector<long> bincount(num_classes_list[o], 0);
                for (unsigned long i = 0; i < num_samples; ++i) {
                    bincount[y[i * num_outputs + o]]++;
                }
                for (unsigned long c = 0; c < num_classes_list[o]; ++c) {
                    class_weight[o * max_num_classes + c] =
                            (static_cast<double>(num_samples) / bincount[c]) / num_classes_list[o];
                }
            }
        }
        else {
            if (class_weight_ptr_ == nullptr) {
                throw std::invalid_argument(
                    "If 'class_balanced' is false, must provide a pointer to a class weight."
                    "Weights associated with classes in the form {weight, weight, ...}."
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
        if (criterion_option_ == "gini") {
            criterion_ = decisiontree::Criterion(num_outputs, 
                                                 num_samples, 
                                                 max_num_classes,
                                                 num_classes_list, 
                                                 class_weight);
        }
        else if (criterion_option_ == "entropy") {
            throw std::runtime_error("Not Implemented");
        }
        else {
            throw std::invalid_argument("Criterion must be either 'gini' or 'entropy'.");
        }

        if (random_state_seed_ == -1) {
            random_state_ = decisiontree::RandomState();
        }
        else {
            random_state_ = decisiontree::RandomState(random_state_seed_);
        }

        decisiontree::Splitter splitter(num_features, 
                                        num_samples,
                                        max_num_features,
                                        split_policy_,
                                        criterion_,
                                        random_state_);
        decisiontree::Tree tree(num_outputs, 
                                num_features, 
                                num_classes_list);

    };







};

} // namespace

#endif // DECISION_TREE_CLASSIFIER_HPP_