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
        std::size_t left_child;
        std::size_t right_child;
        std::size_t feature_indice;
        int has_missing_value;
        DataType threshold;
        DataType impurity;
        DataType improvement;
        HistogramType histogram;

        TreeNode(std::size_t left_child_, 
                 std::size_t right_child_, 
                 std::size_t feature_indice_,
                 int has_missing_value_,
                 DataType threshold_,
                 DataType impurity_,
                 DataType improvement_,
                 HistogramType histogram_): 
            left_child(left_child_), 
            right_child(right_child_), 
            feature_indice(feature_indice_),
            has_missing_value(has_missing_value_),
            threshold(threshold_),
            impurity(impurity_),
            improvement(improvement_),
            histogram(histogram) {};
        
        ~TreeNode() {};
    };

    std::size_t num_outputs_;
    std::size_t num_features_;
    
    std::vector<std::size_t> num_classes_list_;

    std::size_t max_depth_;
    std::size_t node_count_;
    std::size_t max_num_classes_;
    std::vector<TreeNode> nodes_;

public:
    Tree(std::size_t num_outputs, 
         std::size_t num_features, 
         std::vector<std::size_t> num_classes_list): 
            num_outputs_(num_outputs),
            num_features_(num_features),
            num_classes_list_(num_classes_list) {
        
        max_depth_ = 0;
        node_count_ = 0;
        max_num_classes_ = *std::max_element(std::begin(num_classes_list), std::end(num_classes_list));
    };
    ~Tree() {};

    std::size_t add_node(bool is_left,
                         std::size_t depth, 
                         std::size_t parent_indice, 
                         std::size_t feature_indice, 
                         int has_missing_value, 
                         DataType threshold, 
                         DataType impurity, 
                         DataType improvement, 
                         const HistogramType& histogram) {

        TreeNode* cur_node = new TreeNode(
            0, 0, 
            feature_indice,
            has_missing_value,
            threshold,
            impurity,
            improvement,
            histogram);
        nodes_.emplace_back(*cur_node);

        std::size_t node_indice = node_count_++;

        // not root node
        if (depth > 0) {
            if (is_left) {
                nodes_[node_indice].left_child = node_indice;
            }
            else {
                nodes_[node_indice].right_child = node_indice;
            }
        }

        if (depth > max_depth_) {
            max_depth_ = depth;
        }

        return node_indice;
    }


};

} // namespace decision-tree

#endif