#pragma once

#include "source.h"

class TileFactory : public Source {
 public:
  static auto Create(Source* inner_source, uint32_t tile_w,
                     uint32_t tile_h) -> std::unique_ptr<TileFactory>;

  ~TileFactory() override = default;
};
