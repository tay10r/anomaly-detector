#pragma once

#include <pipeline/normalize_filter_config.pb.h>

#include "node.h"

class NormalizeFilter : public Node {
 public:
  static auto Create(std::unique_ptr<Node> child,
                     const pipeline::NormalizeFilterConfig& cfg) -> std::unique_ptr<NormalizeFilter>;

  ~NormalizeFilter() override = default;
};
