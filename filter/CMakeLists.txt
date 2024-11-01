cmake_minimum_required(VERSION 3.14.7)

project(ad_pipeline
  DESCRIPTION "A program for processing images for the purpose of anomaly detection."
  LANGUAGES C CXX
  VERSION 1.0)

find_package(spdlog CONFIG REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Will either:
#   - Read from ZMQ publisher and tile the image, optionally add the UV component, and send it to the requester
#   - Read from a image folder and send it to the requester
add_executable(ad_pipeline
  main.cpp
  node.h
  node.cpp
  image.h
  image.cpp
  zmq_source.h
  zmq_source.cpp
  directory_source.h
  directory_source.cpp
  directory_sink.h
  directory_sink.cpp
  tile_filter.h
  tile_filter.cpp
  normalize_filter.h
  normalize_filter.cpp
  detection_filter.h
  detection_filter.cpp
  frame_builder.h
  frame_builder.cpp
  exception.h
  exception.cpp
  deps/stb_image.h
  deps/stb_image.c
  deps/stb_image_write.h
  deps/stb_image_write.c)

protobuf_generate(TARGET ad_pipeline
  PROTOS
    proto/pipeline/config.proto
    proto/pipeline/detection_filter_config.proto
    proto/pipeline/directory_source_config.proto
    proto/pipeline/directory_sink_config.proto
    proto/pipeline/frame_builder_config.proto
    proto/pipeline/tile_filter_config.proto
    proto/pipeline/normalize_filter_config.proto
    proto/pipeline/zmq_source_config.proto
    proto/pipeline/padding_mode.proto
  IMPORT_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/proto")

target_link_libraries(ad_pipeline
  PUBLIC
    spdlog::spdlog
    libzmq
    libzmq-static
    protobuf::libprotobuf
    ${OpenCV_LIBRARIES}
    OpenMP::OpenMP_CXX)

target_compile_features(ad_pipeline PRIVATE cxx_std_20)

target_include_directories(ad_pipeline
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/deps"
    "${CMAKE_CURRENT_BINARY_DIR}")

set_target_properties(ad_pipeline
  PROPERTIES
    OUTPUT_NAME ad-pipeline)
