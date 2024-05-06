cmake_minimum_required(VERSION 3.10)
project(decision_tree LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas")
set(CMAKE_CXX_FLAGS_RELEASE "-s -O3 -mavx -msse2 -mfma -mbmi -Wno-unused-variable")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g")
set(CMAKE_CXX_FLAGS_SANITIZE "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
set(CMAKE_CXX_FLAGS_SANITIZEREL "${CMAKE_CXX_FLAGS_RELEASE} -g -fsanitize=address")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.4")
    message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
    message(FATAL_ERROR "\decision-tree requires g++ at least v11.4")
  endif()
endif()




