#pragma once

#include <pipeline/tile_filter_config.pb.h>

#include "node.h"

class TileFilter : public Node {
 public:
  static auto Create(std::unique_ptr<Node> child,
                     const pipeline::TileFilterConfig& cfg)
      -> std::unique_ptr<TileFilter>;

  ~TileFilter() override = default;
};
