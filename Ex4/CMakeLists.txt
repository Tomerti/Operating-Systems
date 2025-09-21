cmake_minimum_required(VERSION 3.24)
project(Ex4)

set(CMAKE_CXX_STANDARD 11)

add_executable(Ex4
        VirtualMemory.h
        PhysicalMemory.h
        PhysicalMemory.cpp
        VirtualMemory.cpp)
