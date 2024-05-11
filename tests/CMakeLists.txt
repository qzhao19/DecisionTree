enable_testing()
find_package(GTest REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

add_executable(unittests 
    test_builder.cpp 
    test_criterion_gini.cpp 
    test_math.cpp 
    test_sort.cpp 
    test_splitter.cpp 
    test_tree.cpp)

target_link_libraries(unittests GTest::GTest GTest::Main)

gtest_discover_tests(unittests)