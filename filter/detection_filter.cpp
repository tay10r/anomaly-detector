#include "detection_filter.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <opencv2/dnn.hpp>

#include "exception.h"

namespace {

class DetectionFilterImpl final : public DetectionFilter {
 public:
  DetectionFilterImpl(std::unique_ptr<Node> child_node, const pipeline::DetectionFilterConfig& cfg)
      : child_node_(std::move(child_node)), config_(cfg) {
    if (config_.model().empty()) {
      throw Exception("Model path is empty.");
    }

    if (config_.infill_width() == 0) {
      throw Exception("Infill width cannot be zero.");
    }

    if (config_.infill_height() == 0) {
      throw Exception("Infill height cannot be zero.");
    }

    SPDLOG_INFO("Infill location set to ({}, {}) with area of {}x{} and model path of '{}'.", config_.infill_x(),
                config_.infill_y(), config_.infill_width(), config_.infill_height(), config_.model());
  }

  [[nodiscard]] auto Step() -> NodeOutput override {
    auto child_output = child_node_->Step();
    if (child_output.EndOfStream()) {
      return NodeOutput();
    }

    if (!net_) {
      if (!LoadModel()) {
        return NodeOutput();
      }
    }

    return Process(child_output);
  }

 protected:
  [[nodiscard]] auto LoadModel() -> bool {
    net_.reset(new cv::dnn::Net(cv::dnn::readNetFromONNX(config_.model())));

    if (net_->empty()) {
      SPDLOG_ERROR("Failed to load model '{}'.", config_.model());
      return false;
    }

    SPDLOG_INFO("Loaded model '{}'.", config_.model());

    return true;
  }

  [[nodiscard]] auto CheckShape(const NodeOutput& child_output) -> bool {
    const auto w = child_output.image->Width();
    const auto h = child_output.image->Height();

    const auto max_x = config_.infill_x() + config_.infill_width();
    const auto max_y = config_.infill_y() + config_.infill_height();

    if ((max_x > w) || (max_y > h)) {
      SPDLOG_ERROR("Image is not large enough for infill (need {}x{}, but got {}x{}.", max_x, max_y, w, h);
      return false;
    }

    return true;
  }

  [[nodiscard]] auto CreateInput(Image& img) -> cv::Mat {
    cv::Mat input(img.Height(), img.Width(), CV_8UC3, img.Data());
    return cv::dnn::blobFromImage(input, 1.0 / 255.0);
  }

  [[nodiscard]] auto CheckOutputShape(const cv::Mat& output) -> bool {
    if ((output.rows != config_.infill_height()) || (output.cols != config_.infill_width())) {
      SPDLOG_ERROR("Expected output size of {}x{} but got {}x{}", config_.infill_width(), config_.infill_height(),
                   output.cols, output.rows);
      return false;
    }

    return true;
  }

  template <typename Scalar>
  [[nodiscard]] static constexpr auto Square(Scalar x) -> Scalar {
    return x * x;
  }

  [[nodiscard]] auto CreateOutput(const NodeOutput& child_output, cv::Mat& output) -> NodeOutput {
    auto detection_output = std::make_shared<Image>(output.rows, output.cols);

    const auto num_pixels = output.rows * output.cols;

    const auto* input = child_output.image->Data();
    const auto input_w = child_output.image->Width();

#pragma omp parallel for
    for (auto i = 0; i < num_pixels; i++) {
      const auto x = i % output.cols;
      const auto y = i / output.cols;

      const auto predicted_r = output.at<float>(static_cast<int>(i * 3 + 0)) * 255.0F;
      const auto predicted_g = output.at<float>(static_cast<int>(i * 3 + 1)) * 255.0F;
      const auto predicted_b = output.at<float>(static_cast<int>(i * 3 + 2)) * 255.0F;

      const auto in_offset = ((config_.infill_y() + y) * input_w + (config_.infill_x() + x)) * 3;

      const auto measured_r = static_cast<float>(input[in_offset + 0]);
      const auto measured_g = static_cast<float>(input[in_offset + 1]);
      const auto measured_b = static_cast<float>(input[in_offset + 2]);

      constexpr auto scale{1.0F / 255.0F};
      const auto delta_r = static_cast<int>(Square(predicted_r - measured_r) * scale);
      const auto delta_g = static_cast<int>(Square(predicted_g - measured_g) * scale);
      const auto delta_b = static_cast<int>(Square(predicted_b - measured_b) * scale);

      auto* out = detection_output->Data() + i * 3;
      out[0] = static_cast<std::uint8_t>(std::clamp(delta_r, 0, 255));
      out[1] = static_cast<std::uint8_t>(std::clamp(delta_g, 0, 255));
      out[2] = static_cast<std::uint8_t>(std::clamp(delta_b, 0, 255));
    }

    NodeOutput self_output(detection_output, child_output.frame_id);
    self_output.offset[0] = child_output.offset[0] + config_.infill_x();
    self_output.offset[1] = child_output.offset[1] + config_.infill_y();
    self_output.size = child_output.size;
    return self_output;
  }

  [[nodiscard]] auto Process(const NodeOutput& child_output) -> NodeOutput {
    if (!CheckShape(child_output)) {
      return NodeOutput();
    }
    auto input_blob = CreateInput(*child_output.image);

    net_->setInput(input_blob);

    auto output_blob = net_->forward();

    std::vector<cv::Mat> outputs;

    cv::dnn::imagesFromBlob(output_blob, outputs);

    if (outputs.empty()) {
      SPDLOG_ERROR("Forward pass did not produce any outputs.");
      return NodeOutput();
    }

    if (!CheckOutputShape(outputs[0])) {
      return NodeOutput();
    }

    SPDLOG_INFO("Completed forward pass.");

    return CreateOutput(child_output, outputs.at(0));
  }

 private:
  std::unique_ptr<Node> child_node_;

  pipeline::DetectionFilterConfig config_;

  std::unique_ptr<cv::dnn::Net> net_;
};

}  // namespace

auto DetectionFilter::Create(std::unique_ptr<Node> child_node,
                             const pipeline::DetectionFilterConfig& cfg) -> std::unique_ptr<DetectionFilter> {
  return std::make_unique<DetectionFilterImpl>(std::move(child_node), cfg);
}
