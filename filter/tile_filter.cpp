#include "tile_filter.h"

#include <algorithm>
#include <optional>

namespace {

struct TileState final {
  NodeOutput child_output;
  std::uint32_t x{};
  std::uint32_t y{};
};

class TileFilterImpl final : public TileFilter {
 public:
  TileFilterImpl(std::unique_ptr<Node> child, const pipeline::TileFilterConfig& config)
      : child_(std::move(child)), config_(config) {}

  [[nodiscard]] auto Step() -> NodeOutput override {
    if (!current_state_) {
      current_state_ = TileState{child_->Step()};
    }

    if (current_state_->child_output.EndOfStream()) {
      return current_state_->child_output;
    }

    auto tile = std::make_shared<Image>(config_.width(), config_.height());

    if (tile->Empty() || current_state_->child_output.EndOfStream()) {
      return NodeOutput();
    }

    Blit(tile);

    auto output{NodeOutput(std::move(tile), current_state_->child_output.frame_id)};
    output.offset[0] = current_state_->child_output.offset[0] + current_state_->x;
    output.offset[1] = current_state_->child_output.offset[1] + current_state_->y;
    output.size = current_state_->child_output.size;

    current_state_->x += config_.stride_x();

    if (current_state_->x >= current_state_->child_output.image->Width()) {
      current_state_->x = 0;
      current_state_->y += config_.stride_y();
      if (current_state_->y >= current_state_->child_output.image->Height()) {
        current_state_.reset();
      }
    }

    return output;
  }

 protected:
  void Blit(std::shared_ptr<Image>& tile) {
    const auto w = config_.width();
    const auto h = config_.height();

    auto* dst = tile->Data();

    const auto* src = current_state_->child_output.image->Data();

    const auto replicate{config_.padding_mode() == pipeline::PaddingMode::REPLICATE};

    const auto frame_w = current_state_->child_output.image->Width();
    const auto frame_h = current_state_->child_output.image->Height();

    const auto max_frame_x = replicate ? (frame_w - 1) : frame_w;
    const auto max_frame_y = replicate ? (frame_h - 1) : frame_h;

    for (std::uint32_t y = 0; y < h; y++) {
      const auto src_y = std::clamp(current_state_->y + y, 0u, max_frame_y);

      for (std::uint32_t x = 0; x < w; x++) {
        std::array<uint8_t, 3> rgb{0, 0, 0};

        const auto src_x = std::clamp(current_state_->x + x, 0u, max_frame_x);

        if ((src_x < frame_w) && (src_y < frame_h)) {
          const auto src_i = src_y * current_state_->child_output.image->Width() + src_x;
          rgb[0] = src[src_i * 3 + 0];
          rgb[1] = src[src_i * 3 + 1];
          rgb[2] = src[src_i * 3 + 2];
        }

        const auto dst_i = y * w + x;
        dst[dst_i * 3 + 0] = rgb[0];
        dst[dst_i * 3 + 1] = rgb[1];
        dst[dst_i * 3 + 2] = rgb[2];
      }
    }
  }

 private:
  std::optional<TileState> current_state_;

  std::unique_ptr<Node> child_;

  pipeline::TileFilterConfig config_;
};

}  // namespace

auto TileFilter::Create(std::unique_ptr<Node> child,
                        const pipeline::TileFilterConfig& config) -> std::unique_ptr<TileFilter> {
  return std::make_unique<TileFilterImpl>(std::move(child), config);
}
