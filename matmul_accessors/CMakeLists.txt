cmake_minimum_required(VERSION 3.18)

project(matmul)

set(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE ON)


#add_subdirectory(alpaka)
find_package(alpaka REQUIRED)

alpaka_add_executable(${PROJECT_NAME} matmul.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE alpaka::alpaka)
