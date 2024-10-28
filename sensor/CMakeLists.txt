cmake_minimum_required(VERSION 3.14.7)

project(ad_sensor
  VERSION 1.0
  DESCRIPTION "The sensor component of the anomaly detector system.")

find_package(ZeroMQ CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cxxopts CONFIG REQUIRED)

add_executable(ad-sensor
  main.cpp)

target_link_libraries(ad-sensor
  PUBLIC
    ${OpenCV_LIBRARIES}
    libzmq
    libzmq-static
    spdlog::spdlog
    cxxopts::cxxopts)
