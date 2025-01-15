cmake_minimum_required(VERSION 3.24)
project(gpu-count-smp LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(random_menPrefs src/random_menPrefs.cu)
add_executable(find_womenPref_with_targetCount src/find_womenPref_with_targetCount.cu)
add_executable(bruteforce src/bruteforce.cu)
