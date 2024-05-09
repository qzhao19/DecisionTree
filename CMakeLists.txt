# check CMake version
cmake_minimum_required(VERSION 3.26.5)

# project info
project(decision_tree LANGUAGES CXX VERSION 1.0.0)

option(BUILD_DEMO "Build demo" ON)
option(BUILD_TEST "Build test" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS_RELEASE "-s -O3 -mavx -msse2 -mfma -mbmi -Wno-unused-variable")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g")
# set(CMAKE_CXX_FLAGS_SANITIZE "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
# set(CMAKE_CXX_FLAGS_SANITIZEREL "${CMAKE_CXX_FLAGS_RELEASE} -g -fsanitize=address")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.4")
        message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
        message(FATAL_ERROR "\ndecision-tree requires g++ at least v11.4")
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ./decision_tree)

add_library(dtreelib INTERFACE)
target_sources(dtreelib INTERFACE FILE_SET HEADERS 
               BASE_DIRS ${PROJECT_SOURCE_DIR}
               FILES decision_tree.hpp)
add_subdirectory("decision_tree/algorithm")
add_subdirectory("decision_tree/common")
add_subdirectory("decision_tree/core")
add_subdirectory("decision_tree/core/criterion")
add_subdirectory("decision_tree/utility")

if(BUILD_DEMO)
    file(GLOB DEMO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)
    foreach(demo_file ${DEMO_SOURCES})
        get_filename_component(demo_name ${demo_file} NAME_WE)
        add_executable(${demo_name})
        target_sources(${demo_name} PRIVATE ${demo_file})
        target_link_libraries(${demo_name} PRIVATE dtreelib)
    endforeach(demo_file ${DEMO_SOURCES})
endif()

if(BUILD_TEST)
    add_subdirectory(tests)
endif()
