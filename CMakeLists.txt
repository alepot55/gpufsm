cmake_minimum_required(VERSION 3.18)
project(gpufsm LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Graceful CUDA build.
# The CUDA pybind11 extension is OPTIONAL. With -DGPUFSM_BUILD_CUDA=ON and a
# working CUDA toolkit it is compiled and installed into the package; otherwise
# the build is a no-op and the pure-Python package (CPU + Triton) still works.
# -----------------------------------------------------------------------------
option(GPUFSM_BUILD_CUDA "Build the CUDA pybind11 extension" OFF)

if(NOT GPUFSM_BUILD_CUDA)
    message(STATUS "gpufsm: GPUFSM_BUILD_CUDA=OFF -> skipping CUDA extension (pure-Python install).")
    return()
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "gpufsm: CUDA requested but no CUDA compiler found -> skipping CUDA extension.")
    return()
endif()

enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB_RECURSE GPUFSM_CUDA_SOURCES CONFIGURE_DEPENDS
    "src/gpufsm/backends/cuda/*.cu"
    "src/gpufsm/backends/cuda/*.cpp")

if(NOT GPUFSM_CUDA_SOURCES)
    message(WARNING "gpufsm: no CUDA sources found yet under src/gpufsm/backends/cuda -> skipping.")
    return()
endif()

pybind11_add_module(_cuda ${GPUFSM_CUDA_SOURCES})
target_include_directories(_cuda PRIVATE src/gpufsm/backends/cuda/include)
target_compile_options(_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --extended-lambda --expt-relaxed-constexpr>)
install(TARGETS _cuda DESTINATION gpufsm/backends/cuda)
