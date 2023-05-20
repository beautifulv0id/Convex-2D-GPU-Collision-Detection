cmake_minimum_required(VERSION 3.10)
project(dcpf_generate)

include_directories(/usr/include)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(dcpf_generate sat_cpu.cpp)

target_link_libraries(dcpf_generate PRIVATE OpenMP::OpenMP_CXX)
