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
 * @file decision_tree_class.hpp
 * 
 * @brief A decision tree classifier.
 * 
 * @param feature_names 1d array of string
 *  which refers names to use for elements of input data given.
 * @param class_labels 2d array of shape (num_outputs, num_classes) 
 *  which is 2d arrays of class labels, e.g {{<class1>, <class2>, ...}}
 * @param random_seed int default=0
 *  controls the randomness of the estimator
 * @param max_depth int default=4,
 *  the maximum depth of the tree
 * @param max_num_features int default=-1,
 *  the number of features to consider when looking for the best split
 *  if -1, then max_num_features = num_features, else consider value 
 *  given by the user.
 * @param min_samples_split int default=2
 *  the minimum number of samples required to split an internal node.
 * @param min_samples_leaf int default=1
 *  the minimum number of samples required to be at a leaf node.
 * @param min_weight_fraction_leaf double default=0.0
 *  the minimum weighted fraction of the sum total of weights required to be at a leaf node
 * @param class_balanced boolean default=true
 *  indicate whether the class is balanced or not, if set to true, it uses the values 
 *  of y to automatically adjust weights inversely proportional to class frequencies 
 *  in the input data as n_samples / (n_classes * bincount(y)), else the user must provide
 *  a self-defined class weights
 * @param criterion {"gini", "entropy"} default=gini
 *  the criterion to measure the quality of a split
 * @param split_policy {"random", "best"} default=best
 *  the strategy used to choose the split at each node.
 * @param class_weight_ptr_ smart pointer to the class weight
 *  if class_balanced=false, class_weight_ptr_ must be defined to
 *  point to the self-defined class weight in the form {w, w, ..., }
*/
class DecisionTreeClassifier {
private:
    std::vector<std::string> feature_names_;
    std::vector<std::vector<std::string>> class_labels_;
    int random_seed_;
    int max_depth_;
    int max_num_features_;
    int min_samples_split_;
    int min_samples_leaf_;
    double min_weight_fraction_leaf_;
    bool class_balanced_;
    std::string criterion_;
    std::string split_policy_;
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
                        random_seed_(random_seed),
                        max_depth_(max_depth),
                        max_num_features_(max_num_features),
                        min_samples_split_(min_samples_split),
                        min_samples_leaf_(min_samples_leaf),
                        min_weight_fraction_leaf_(min_weight_fraction_leaf),
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

        // check split strategy
        if (SPLIT_STRATEGY.find(split_policy_) == SPLIT_STRATEGY.end()) {
            throw std::invalid_argument(
                "Supported strategies are 'best' to choose the best "
                "split and 'random' to choose the best random split.");
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