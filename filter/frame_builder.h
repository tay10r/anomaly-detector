#pragma once

#include <pipeline/frame_builder_config.pb.h>

#include <memory>

#include "node.h"

class FrameBuilder : public Node {
 public:
  static auto Create(std::unique_ptr<Node> child_node,
                     const pipeline::FrameBuilderConfig&) -> std::unique_ptr<FrameBuilder>;

  ~FrameBuilder() override = default;
};
