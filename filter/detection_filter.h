#pragma once

#include <pipeline/detection_filter_config.pb.h>

#include <memory>

#include "node.h"

class DetectionFilter : public Node {
 public:
  static auto Create(std::unique_ptr<Node> child,
                     const pipeline::DetectionFilterConfig& cfg) -> std::unique_ptr<DetectionFilter>;

  ~DetectionFilter() override = default;
};
