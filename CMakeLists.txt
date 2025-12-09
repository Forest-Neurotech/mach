cmake_minimum_required(VERSION 3.18...3.27)
# Override the project name to "mach" so we can import mach, even though the wheel is called "mach-beamform"
set(SKBUILD_PROJECT_NAME "mach")
# Prefer g++ compiler for C++ code
set(CMAKE_CXX_COMPILER g++)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CUDA CXX)

# https://github.com/wjakob/nanobind_example/blob/1c2a122/CMakeLists.txt
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
    pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
    pip install nanobind scikit-build-core[pyproject]
    pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()


# search for Python >= 3.10 including the Development.Module component required by nanobind
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

# set the build type to Release by default
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# find the nanobind package
find_package(nanobind CONFIG REQUIRED)

# CUDA requires C++11 or newer
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_INIT} ${CMAKE_CUDA_FLAGS}")
# includes details about register usage and shared memory for each kernel.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
# debug info
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
# Compute-capability 7.5+, to support Github T4 runner
# https://developer.nvidia.com/cuda-gpus
set(CMAKE_CUDA_ARCHITECTURES 75)

# nanobind requires GCC 8+
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
        message(FATAL_ERROR "g++ version must be at least 8.0 for nanobind!")
    endif()
endif()

# Check minimum CUDA compiler version
set(NVCC_MIN_VERSION 11.0)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${NVCC_MIN_VERSION})
    message(FATAL_ERROR "NVCC (CUDA compiler) version must be at least ${NVCC_MIN_VERSION}!")
endif()

# Parse NVCC version to get major.minor components
set(NVCC_VERSION_STR "${CMAKE_CUDA_COMPILER_VERSION}")
string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" _ "${NVCC_VERSION_STR}")
set(NVCC_MAJOR ${CMAKE_MATCH_1})
set(NVCC_MINOR ${CMAKE_MATCH_2})

message(STATUS "Parsed NVCC version: major=${NVCC_MAJOR}, minor=${NVCC_MINOR}")

# Validate parsing worked
if(NOT DEFINED NVCC_MAJOR OR NOT DEFINED NVCC_MINOR)
    message(FATAL_ERROR "Failed to parse NVCC version '${NVCC_VERSION_STR}'. Expected format: X.Y")
endif()

# Add vendor directory to include paths
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/${SKBUILD_PROJECT_NAME}/vendor)

# Build the nanobind module with the CUDA kernel
nanobind_add_module(
  # Name of the extension
  _cuda_impl

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  NB_STATIC

  # Source code
  src/${SKBUILD_PROJECT_NAME}/kernel.cu
)

# Pass CUDA version information as compiler definitions
target_compile_definitions(_cuda_impl PRIVATE
    NVCC_VERSION_STR="${NVCC_VERSION_STR}"
    NVCC_MAJOR=${NVCC_MAJOR}
    NVCC_MINOR=${NVCC_MINOR}
)

# install the CUDA module
install(TARGETS _cuda_impl LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
