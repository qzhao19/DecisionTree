## DecisionTree

[![C++14](https://img.shields.io/badge/C%2B%2B-14-blue.svg)](https://isocpp.org/std/the-standard) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The library is an implementation of the decision tree classification algorithm in C++. It is designed to be a robust tool for classification tasks within a supervised learning context.


## Overview

Decision Tree (DT) is a powerful supervised learning method for classification and regression. They offer several compelling benefits, such as their simplicity of understanding and interpretation, and the minimal data preparation they require. The `decisiontree` library is crafted to provide clean and intuitive interfaces, ensuring ease of use and seamless integration into your projects.

## Key Features

- **Ease of Integration**: Simple interfaces for hassle-free incorporation into existing projects.
- **Simplicity**: Straightforward logic that is easy to apply.
- **Versatility**: Effective for both classification problems.


## System Requirements

- A C++ compiler that supports C++14 or later.
- CMake version 3.26.5 or higher.
- Google Test (GTest) version 1.14.0 or higher for running unit tests.

## Installation Guide

To install the `decisiontree` library, follow these manual steps using CMake:

1. Clone or download and unzip the repository.
2. Navigate to the repository's root directory.
3. Execute the following commands:

```bash
cmake -S . -B build
cmake --build build
sudo make install  # Use 'sudo' if administrative privileges are required for installation.
```

This process will configure, build, and install the library on your system, making it ready for use in your C++ projects.

## Example

We also provide three examples to demonstrate how to use the `decisiontree` library, see [example1.cpp](https://github.com/qzhao19/decision-tree/blob/main/examples/example1.cpp).


Ensure you have installed the decisiontree library and that it is available in your system's include path. Compile and run the example using the following command:

```bash
g++ -std=c++14 -I/path/to/decisiontree/include -o example1.out example1.cpp
./example1.out
```


