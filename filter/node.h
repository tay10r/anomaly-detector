#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <memory>

#include "image.h"

/**
 * @brief Contains the output data of a node.
 *
 * @note Source nodes are nodes who do not have a child node. In other words,
 * source nodes can be thought of as input nodes. Generally speaking, source
 * nodes include things like image directories or video streams. All other nodes
 *       that are not source nodes are called filter nodes.
 *
 * @note In general, a node output contains an image. In some cases, the image
 * may be a smaller section of larger one. The original images from the sensor
 * are called frames. Smaller sub-sections of a frame are called tiles. By
 *       default, the first source node has a tile size equal to the size of the
 * frame.
 * */
struct NodeOutput final {
  /**
   * @brief The image created by the node.
   *
   * @details For frames, this is the entire image. For tiles, this is the image
   * data for the tile.
   * */
  std::shared_ptr<Image> image;

  /**
   * @brief The offset of the image from the original frame.
   * */
  std::array<std::uint32_t, 2> offset;

  /**
   * @brief The size of the original frame.
   *
   * @note This may be equal to or larger than the image size.
   * */
  std::array<std::uint32_t, 2> size;

  /**
   * @brief The unique frame ID.
   * */
  std::uint32_t frame_id{std::numeric_limits<std::uint32_t>::max()};

  NodeOutput() = default;

  NodeOutput(std::shared_ptr<Image> img, uint32_t frame_id_)
      : image(std::move(img)), offset{0, 0}, size{image->Width(), image->Height()}, frame_id(frame_id_) {}

  NodeOutput(std::shared_ptr<Image> img, const NodeOutput& child)
      : image(std::move(img)), offset(child.offset), size(child.size), frame_id(child.frame_id) {}

  [[nodiscard]] auto EndOfStream() const -> bool { return frame_id == std::numeric_limits<std::uint32_t>::max(); }
};

class Node {
 public:
  static auto CreatePipeline(void* zmq_context, const char* config_path) -> std::unique_ptr<Node>;

  virtual ~Node() = default;

  [[nodiscard]] virtual auto Step() -> NodeOutput = 0;
};
