cmake_minimum_required(VERSION 3.10)

project(trans_pearl VERSION 0.0.1)

include_directories(include ${Eigen_INCLUDE_DIRS})

set(sourcefiles
src/trans_pearl_bindings.cpp
src/trans_pearl.cpp
src/trans_pearl_wrapper.cpp
)

set(include_dirs
src/
third_party/
)

set(glog "-lglog -lpthread")
set(warning "-Wall -Wextra -Wno-sign-compare -Wno-nonnull -Wno-unused-variable")
set(flags "-fPIC -std=c++14 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE64 -O0 -DUNIX -g")

add_definitions(${flags})

set(ADD_PEARL_AS_LIBRARY ON)

add_subdirectory(third_party/PEARL)
add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/knn-cpp)
pybind11_add_module(trans_pearl_wrapper SHARED ${sourcefiles})

target_link_libraries(trans_pearl_wrapper PUBLIC pearl)
target_include_directories(trans_pearl_wrapper PUBLIC ${include_dirs})