#ifndef UTILITY_SORT_HPP
#define UTILITY_SORT_HPP

namespace decisiontree {

template <typename DataType, typename IndexType>
void sort(std::vector<DataType>& x, 
          std::vector<IndexType>& y, 
          std::size_t start, 
          std::size_t end,
          bool reverse=false) {
    
    if (x.size() != y.size()) {
        throw std::out_of_range("Size of two vector should be equal.");
    }

    // combine x and y into vector pair<x, y>
    std::vector<std::pair<DataType, IndexType>> combine(x.size());
    auto x_iter = x.begin();
    auto y_iter = y.begin();
    for (auto& xy : combine) {
        xy.first = *(x_iter++);
        xy.second = *(y_iter++);
    }

    // sort x_y within the range [start:end]
    if (!reverse) {
        sort(combine.begin() + start, 
             combine.end() - (combine.size() - end), 
             [](const std::pair<DataType, IndexType>& left, 
                const std::pair<DataType, IndexType>& right) -> bool {
                    return left.first < right.first;
            });
    }
    else {
        sort(combine.begin() + start, 
             combine.end() - (combine.size() - end), 
             [](const std::pair<DataType, IndexType>& left, 
                const std::pair<DataType, IndexType>& right) -> bool {
                    return left.first > right.first;
            });
    }

    // copy sorted vector pair<x,y> back into x, y
    x_iter = x.begin();
    y_iter = y.begin();
    for (auto& xy : combine) {
        *(x_iter++) = xy.first;
        *(y_iter++) = xy.second;
    }
};

} // namespace 
#endif // UTILITY_SORT_HPP