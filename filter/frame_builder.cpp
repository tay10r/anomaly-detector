#include "frame_builder.h"

#include <optional>
#include <vector>

#include <cstring>

namespace {

class FrameBuilderImpl final : public FrameBuilder {
 public:
  explicit FrameBuilderImpl(std::unique_ptr<Node> child_node, const pipeline::FrameBuilderConfig&)
      : child_node_(std::move(child_node)) {}

  [[nodiscard]] auto Step() -> NodeOutput override {
    std::optional<NodeOutput> self_output;

    while (true) {
      NodeOutput child_output;

      if (!child_outputs_.empty()) {
        child_output = std::move(child_outputs_[0]);
        child_outputs_.erase(child_outputs_.begin());
      } else {
        child_output = child_node_->Step();
      }

      if (child_output.EndOfStream()) {
        break;
      }

      if (!self_output) {
        self_output =
            NodeOutput(std::make_shared<Image>(child_output.size[0], child_output.size[1]), child_output.frame_id);
        self_output->offset[0] = 0;
        self_output->offset[1] = 0;
        std::memset(self_output->image->Data(), 0, self_output->image->Width() * self_output->image->Height() * 3);
      } else if (self_output->frame_id != child_output.frame_id) {
        // The child node has finished the current frame and moved to the next.
        // Queue the child output for the next frame and return the completed frame to the parent node.
        child_outputs_.emplace_back(std::move(child_output));
        break;
      }

      auto& frame = *self_output->image;

      const auto& tile = *child_output.image;

      for (auto y = 0; y < tile.Height(); y++) {
        const auto dst_offset = (((y + child_output.offset[1]) * frame.Width()) + child_output.offset[0]) * 3;
        std::memcpy(frame.Data() + dst_offset, tile.Data() + y * tile.Width() * 3, 3 * tile.Width());
      }
    }

    if (!self_output) {
      return NodeOutput();
    }

    return self_output.value();
  }

 private:
  std::vector<NodeOutput> child_outputs_;

  std::unique_ptr<Node> child_node_;
};

}  // namespace

auto FrameBuilder::Create(std::unique_ptr<Node> child_node,
                          const pipeline::FrameBuilderConfig& cfg) -> std::unique_ptr<FrameBuilder> {
  return std::make_unique<FrameBuilderImpl>(std::move(child_node), cfg);
}
