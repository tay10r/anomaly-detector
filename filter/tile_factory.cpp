#include "tile_factory.h"

namespace {

class TileFactoryImpl final : public TileFactory {
 public:
};

}  // namespace

auto TileFactory::Create(Source* inner_source, uint32_t tile_w,
                         uint32_t tile_h) -> std::unique_ptr<TileFactory> {}
