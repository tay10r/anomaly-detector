cmake_minimum_required(VERSION 3.14.7)

project(loader
  DESCRIPTION "A program for pre-processing images before being analyzed for anomaly detection."
  LANGUAGES C CXX
  VERSION 1.0)

find_package(spdlog CONFIG REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

# Will either:
#   - Read from ZMQ publisher and tile the image, optionally add the UV component, and send it to the requester
#   - Read from a image folder and send it to the requester
add_executable(loader
  main.cpp
  source.h
  source.cpp
  image.h
  image.cpp
  zmq_source.h
  zmq_source.cpp
  directory_source.h
  directory_source.cpp
  exception.h
  deps/stb_image.h
  deps/stb_image.c
  deps/stb_image_write.h
  deps/stb_image_write.c)

#file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/loader")

protobuf_generate(TARGET loader
  PROTOS
    proto/loader/config.proto
    proto/loader/directory_config.proto
  IMPORT_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/proto")

target_link_libraries(loader
  PUBLIC
    spdlog::spdlog
    libzmq
    libzmq-static
    protobuf::libprotobuf)

target_include_directories(loader
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/deps"
    "${CMAKE_CURRENT_BINARY_DIR}")

set_target_properties(loader
  PROPERTIES
    OUTPUT_NAME main)
