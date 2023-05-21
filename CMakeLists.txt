#cmake_minimum_required(VERSION 3.17)
#project(edsr-ffmpeg)
#
#set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
#include_directories(/home/anil/libtorch/include)
#include_directories(/home/anil/libtorch/include/torch/csrc/api/include)
#
#find_package( Torch REQUIRED )
#
#add_executable(libtorch_test main.cpp pytorch_model.cpp)



cmake_minimum_required(VERSION 3.17)
project(edsr-ffmpeg)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_PREFIX_PATH "/home/anil/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")

set(libSRCS pytorch_model.cpp pytorch_model.cc)

#add_executable(libtorch_test main.cpp pytorch_model.cpp)
add_library(edsr SHARED ${libSRCS})

target_link_libraries(edsr "${TORCH_LIBRARIES}")
