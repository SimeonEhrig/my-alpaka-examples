CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)
PROJECT(constantMem)

set(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON)

#set(ALPAKA_ACC_GPU_CUDA_ENABLE ON)
#set(ALPAKA_ACC_GPU_HIP_ENABLE OFF)
set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures")

FIND_PACKAGE(alpaka REQUIRED)

ALPAKA_ADD_EXECUTABLE(
  ${CMAKE_PROJECT_NAME}
  main.cpp)

TARGET_LINK_LIBRARIES(
  ${CMAKE_PROJECT_NAME}
  PUBLIC alpaka::alpaka)
