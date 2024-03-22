#ifndef CORE_TREE_HPP
#define CORE_TREE_HPP

#include "prereqs.hpp"

namespace decisiontree {

/**
 * The binary tree is represented as a number of parallel arrays. The i-th
 * element of each array holds information about the node `i`. Node 0 is the
 * tree's root. 
 * 
 * For node data stored at index i, the two child nodes are at 
 * index (2 * i + 1) and (2 * i + 2); the parent node is (i - 1) // 2 
 * (where // indicates integer division).
*/
class Tree {
private:
    
    struct TreeNode {
        NodeIndexType left_child;
        NodeIndexType right_child;
        FeatureIndexType feature_index;
        int has_missing_value;
        FeatureType threshold;
        double impurity;
        double improvement;
        std::vector<std::vector<HistogramType>> histogram;

        TreeNode(NodeIndexType left_child_, 
                 NodeIndexType right_child_, 
                 FeatureIndexType feature_index_,
                 int has_missing_value_,
                 FeatureType threshold_,
                 double impurity_,
                 double improvement_,
                 std::vector<std::vector<HistogramType>> histogram_): 
            left_child(left_child_), 
            right_child(right_child_), 
            feature_index(feature_index_),
            has_missing_value(has_missing_value_),
            threshold(threshold_),
            impurity(impurity_),
            improvement(improvement_),
            histogram(histogram) {};
        
        ~TreeNode() {};
    };

    NumOutputsType num_outputs_;
    NumFeaturesType num_features_;
    std::vector<NumClassesType> num_classes_list_;

    TreeDepthType max_depth_;
    NodeIndexType node_count_;
    NumClassesType max_num_classes_;
    std::vector<TreeNode> nodes_;

public:
    Tree(NumOutputsType num_outputs, 
         NumFeaturesType num_features, 
         std::vector<NumClassesType> num_classes_list): 
            num_outputs_(num_outputs),
            num_features_(num_features),
            num_classes_list_(num_classes_list) {
        
        max_depth_ = 0;
        node_count_ = 0;
        max_num_classes_ = *std::max_element(std::begin(num_classes_list), std::end(num_classes_list));
    };
    ~Tree() {};

    NodeIndexType add_node(bool is_left,
                         std::size_t depth, 
                         std::size_t parent_index, 
                         std::size_t feature_index, 
                         int has_missing_value, 
                         FeatureType threshold, 
                         double impurity, 
                         double improvement, 
                         const std::vector<std::vector<HistogramType>>& histogram) {

        TreeNode* cur_node = new TreeNode(
            0, 0, 
            feature_index,
            has_missing_value,
            threshold,
            impurity,
            improvement,
            histogram);
        nodes_.emplace_back(*cur_node);

        NodeIndexType node_index = node_count_++;

        // not root node
        if (depth > 0) {
            if (is_left) {
                nodes_[node_index].left_child = node_index;
            }
            else {
                nodes_[node_index].right_child = node_index;
            }
        }

        if (depth > max_depth_) {
            max_depth_ = depth;
        }

        return node_index;
    }


};

} // namespace decision-tree

#endif