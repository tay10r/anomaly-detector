cmake_minimum_required(VERSION 3.18)

project(ad_augmentation
  DESCRIPTION "Data augmentation for reconstruction-based anomaly detection."
  VERSION 1.0
  LANGUAGES CXX C)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

FetchContent_Declare(pybind11 URL "https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip")
FetchContent_MakeAvailable(pybind11)

find_package(OpenMP REQUIRED COMPONENTS CXX)

pybind11_add_module(augmentation
  main.cpp
  deps/stb_image.h
  deps/stb_image.c
  deps/stb_image_write.h
  deps/stb_image_write.c)

target_compile_features(augmentation PUBLIC cxx_std_20)

target_link_libraries(augmentation PUBLIC OpenMP::OpenMP_CXX)

install(TARGETS augmentation DESTINATION .)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  find_package(Python REQUIRED COMPONENTS Interpreter)
  add_custom_target(augmentation_test
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=$<TARGET_FILE_DIR:augmentation>" $<TARGET_FILE:Python::Interpreter> test/main.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
