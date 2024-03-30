#ifndef CORE_BUILDER_HPP_
#define CORE_BUILDER_HPP_

#include "../common/prereqs.hpp"
#include "../utils/random.hpp"

#include "criterion.hpp"
#include "splitter.hpp"
#include "tree.hpp"

namespace decisiontree {

/**
 * @brief build a binary decision tree in depth-first order.
*/
class DepthFirstTreeBuilder {
private:
    TreeDepthType max_depth_;
    NumSamplesType min_samples_split_;
    NumSamplesType min_samples_leaf_;
    ClassWeightType min_weight_leaf_;
    std::vector<ClassWeightType> class_weight_;
    Splitter splitter_;
    
    // node information stack
    // node_info_stk = [start, end, depth, parent_index, is_left]
    struct NodeInfo {
        SampleIndexType start;
        SampleIndexType end;
        TreeDepthType depth;
        NodeIndexType parent_index;
        bool is_left;

        NodeInfo(SampleIndexType start, 
                 SampleIndexType end, 
                 TreeDepthType depth, 
                 NodeIndexType parent_index, 
                 bool is_left): start(start), 
            end(end), 
            depth(depth), 
            parent_index(parent_index), 
            is_left(is_left) {};
        ~NodeInfo() {};
    };

public:
    Tree tree_;
    DepthFirstTreeBuilder(TreeDepthType max_depth, 
                          NumSamplesType min_samples_split, 
                          NumSamplesType min_samples_leaf, 
                          ClassWeightType min_weight_leaf, 
                          std::vector<ClassWeightType> class_weight, 
                          const Splitter& splitter, 
                          const Tree& tree): max_depth_(max_depth), 
                    min_samples_split_(min_samples_split), 
                    min_samples_leaf_(min_samples_leaf), 
                    min_weight_leaf_(min_weight_leaf), 
                    class_weight_(class_weight), 
                    splitter_(splitter), 
                    tree_(tree) {};

    ~DepthFirstTreeBuilder() {};

    void build(const std::vector<FeatureType>& X, 
               const std::vector<ClassType>& y, 
               NumSamplesType num_samples) {
        
        std::stack<NodeInfo> node_info_stk;
        // push root node info into stack
        node_info_stk.push(NodeInfo(0, num_samples, 0, 0, false));

        // allocate a memory of size (2**((max_depth_ + 1))) - 1
        // init node list of tree
        tree_.nodes_.reserve((1u << (max_depth_ + 1)) - 1);

        while(!node_info_stk.empty()) {
            NodeInfo node_info = node_info_stk.top();
            node_info_stk.pop();

            // init weighted class histogram and inpurity for the current node
            splitter_.init_node(y, node_info.start, node_info.end);
            std::vector<std::vector<HistogramType>> histogram = splitter_.criterion_.get_node_weighted_histogram();
            double impurity = splitter_.criterion_.get_node_impurity();

            // get number of samples at the current node
            NumSamplesType num_node_samples = node_info.end - node_info.start;

            bool is_leaf = (node_info.depth >= max_depth_) ||
                            // (num_node_samples < min_samples_split_) || 
                            // (num_node_samples < 2 * min_samples_leaf_) || 
                            // (num_node_samples < 2 * min_weight_leaf_) ||
                            (impurity <= EPSILON);

            FeatureIndexType feature_index = 0;
            SampleIndexType partition_index = 0;
            FeatureType partition_threshold = std::numeric_limits<double>::quiet_NaN();
            double improvement = 0.0;
            int has_missing_value = -1;

            // if not leaf node, split samples[start:end]
            if (!is_leaf) {
                splitter_.split_node(X, y, 
                                     feature_index, 
                                     partition_index, 
                                     partition_threshold, 
                                     improvement, 
                                     has_missing_value);
                if (improvement <= EPSILON) {
                    is_leaf = true;
                }
            }
            // add node to tree
            NodeIndexType node_index = tree_.add_node(node_info.is_left, 
                                                      node_info.depth, 
                                                      node_info.parent_index, 
                                                      feature_index, 
                                                      has_missing_value, 
                                                      partition_threshold, 
                                                      impurity, improvement, 
                                                      histogram);
            
            if (!is_leaf) {
                // push right child node info into the stack
                node_info_stk.emplace(NodeInfo(partition_index, node_info.end, node_info.depth + 1, node_index, false));

                // push left child node info into the stack
                node_info_stk.emplace(NodeInfo(node_info.start, partition_index, node_info.depth + 1, node_index, true));
            }
        };

        tree_.nodes_.shrink_to_fit();

    };

};

} //namespace

#endif // CORE_BUILDER_HPP_