#include "normalize_filter.h"

#include <cmath>

namespace {

class NormalizeFilterImpl final : public NormalizeFilter {
 public:
  NormalizeFilterImpl(std::unique_ptr<Node> child, const pipeline::NormalizeFilterConfig& cfg)
      : child_(std::move(child)), config_(cfg) {}

  [[nodiscard]] auto Step() -> NodeOutput {
    auto child_output = child_->Step();
    if (child_output.EndOfStream()) {
      return child_output;
    }

    auto output_img = std::make_shared<Image>(child_output.image->Width(), child_output.image->Height());

    switch (config_.kind()) {
      case pipeline::Normalization::STANDARD:
        NormalizeStandard(*child_output.image, *output_img);
        break;
      case pipeline::Normalization::MIN_MAX:
        NormalizeMinMax(*child_output.image, *output_img);
        break;
    }

    return NodeOutput(std::move(output_img), child_output);
  }

 protected:
  static auto Sum(const Image& img) -> float {
    auto sum{0.0F};
    const auto num_pixels = img.Width() * img.Height() * 3;
    const auto* data = img.Data();
    for (std::size_t i = 0; i < num_pixels; i++) {
      sum += static_cast<float>(data[i]);
    }
    return sum;
  }

  static auto Stddev(const Image& img, const float avg) -> float {
    auto sum{0.0F};
    const auto num_pixels = img.Width() * img.Height() * 3;
    const auto* data = img.Data();
    for (std::size_t i = 0; i < num_pixels; i++) {
      const auto delta = static_cast<float>(data[i]) - avg;
      sum += delta * delta;
    }
    return std::sqrt(sum / static_cast<float>(num_pixels));
  }

  static void NormalizeStandard(const Image& input, Image& output) {
    const auto sum = Sum(input);
    const auto avg = sum / static_cast<float>(input.Width() * input.Height() * 3);
    const auto stddev = Stddev(input, avg);
    const auto num_pixels = input.Width() * input.Height() * 3;
    auto* dst = output.Data();
    const auto* src = input.Data();
    const auto scale = 1.0F / stddev;
    for (std::size_t i = 0; i < num_pixels; i++) {
      const auto value = static_cast<int>((((static_cast<float>(src[i]) - avg) * scale) + 1.0F) * 0.5F * 255.0F);
      dst[i] = static_cast<std::uint8_t>(std::clamp(value, 0, 255));
    }
  }

  static void NormalizeMinMax(const Image& input, Image& output) {
    auto min_v{255};
    auto max_v{0};
    const auto num_pixels = input.Width() * input.Height() * 3;
    const auto* data = input.Data();
    for (std::size_t i = 0; i < num_pixels; i++) {
      min_v = std::min(min_v, static_cast<int>(data[i]));
      max_v = std::max(max_v, static_cast<int>(data[i]));
    }

    const auto scale = (max_v == min_v) ? 255.0F : (255.0F / static_cast<float>(max_v - min_v));

    auto* dst = output.Data();
    for (std::size_t i = 0; i < num_pixels; i++) {
      const auto value = static_cast<int>((static_cast<float>(data[i]) - static_cast<float>(min_v)) * scale);
      dst[i] = static_cast<std::uint8_t>(std::clamp(value, 0, 255));
    }
  }

 private:
  std::unique_ptr<Node> child_;

  pipeline::NormalizeFilterConfig config_;
};

}  // namespace

auto NormalizeFilter::Create(std::unique_ptr<Node> child,
                             const pipeline::NormalizeFilterConfig& config) -> std::unique_ptr<NormalizeFilter> {
  return std::make_unique<NormalizeFilterImpl>(std::move(child), config);
}
