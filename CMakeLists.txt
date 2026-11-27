cmake_minimum_required(VERSION 3.22)
project(global_sweep LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(MPI_ASSUME_NO_BUILTIN_MPI TRUE)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

set(CSTONE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../gromacs/src/external/exafmm/external/cornerstone-octree"
    CACHE PATH "GROMACS Cornerstone checkout")
set(CSTONE_INCLUDE_DIR "${CSTONE_SOURCE_DIR}/include")

add_executable(test_global_exchange test/test_global_exchange.cpp)
add_executable(test_global_exchange_gpu
    test/test_global_exchange_gpu.cu
    test/device_vector.cu
    ${CSTONE_INCLUDE_DIR}/cstone/primitives/primitives_gpu.cu)

foreach(target test_global_exchange test_global_exchange_gpu)
    target_include_directories(${target} PRIVATE include "${CSTONE_INCLUDE_DIR}")
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
endforeach()

set(NVSHMEM_HOME /user-environment/env/default CACHE PATH "NVSHMEM install prefix")
file(GLOB GTL_LIB_DIR /user-environment/linux-neoverse_v2/cray-gtl-*/lib) # not in the uenv view
target_include_directories(test_global_exchange_gpu PRIVATE "${NVSHMEM_HOME}/include")
target_link_directories(test_global_exchange_gpu PRIVATE "${NVSHMEM_HOME}/lib" ${GTL_LIB_DIR})
target_link_libraries(test_global_exchange_gpu PRIVATE nvshmem_host nvshmem_device nccl mpi_gtl_cuda)
target_link_options(test_global_exchange_gpu PRIVATE "LINKER:-rpath=${GTL_LIB_DIR}")
set_target_properties(test_global_exchange_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(test_global_exchange PRIVATE -Wall -Wextra)
target_compile_options(test_global_exchange_gpu PRIVATE -Xcompiler=-fopenmp)
