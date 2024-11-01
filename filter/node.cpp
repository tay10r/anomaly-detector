#include "node.h"

#include <google/protobuf/util/json_util.h>
#include <pipeline/config.pb.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <fstream>
#include <opencv2/core/utils/logger.hpp>
#include <string>

#include "detection_filter.h"
#include "directory_sink.h"
#include "directory_source.h"
#include "exception.h"
#include "frame_builder.h"
#include "normalize_filter.h"
#include "tile_filter.h"
#include "zmq_sink.h"
#include "zmq_source.h"

namespace {

class NullNode final : public Node {
 public:
  auto Step() -> NodeOutput override { return NodeOutput(); }
};

}  // namespace

auto Node::CreatePipeline(void* zmq_context, const char* config_path) -> std::unique_ptr<Node> {
  std::ifstream file(config_path);
  if (!file.good()) {
    throw Exception("failed to open file");
  }
  file.seekg(0, std::ios::end);
  const auto file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (file_size == -1l) {
    throw Exception("failed to get file size");
  }
  const auto size = static_cast<std::size_t>(file_size);
  std::string data;
  data.resize(size);
  file.read(&data[0], data.size());
  data.resize(file.gcount());

  pipeline::Config config;

  const auto status = google::protobuf::util::JsonStringToMessage(data, &config);
  if (!status.ok()) {
    throw Exception(std::string(status.message()));
  }

  if (!config.enable_cv_logging()) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
  }

  std::unique_ptr<Node> root{new NullNode()};

  for (const auto& node_config : config.pipeline()) {
    std::unique_ptr<Node> node;
    switch (node_config.root_case()) {
      case pipeline::NodeConfig::kZmqSource:
        SPDLOG_INFO("Building ZMQ source node.");
        root = ZmqSource::Create(zmq_context, node_config.zmq_source());
        break;
      case pipeline::NodeConfig::kDirectorySource:
        SPDLOG_INFO("Building directory source node.");
        root = DirectorySource::Create(node_config.directory_source());
        break;
      case pipeline::NodeConfig::kDirectorySink:
        SPDLOG_INFO("Building directory sink node.");
        root = DirectorySink::Create(std::move(root), node_config.directory_sink());
        break;
      case pipeline::NodeConfig::kTileFilter:
        SPDLOG_INFO("Building directory tile filter node.");
        root = TileFilter::Create(std::move(root), node_config.tile_filter());
        break;
      case pipeline::NodeConfig::kNormalizeFilter:
        SPDLOG_INFO("Building normalization filter node.");
        root = NormalizeFilter::Create(std::move(root), node_config.normalize_filter());
        break;
      case pipeline::NodeConfig::kDetectionFilter:
        SPDLOG_INFO("Building detection filter node.");
        root = DetectionFilter::Create(std::move(root), node_config.detection_filter());
        break;
      case pipeline::NodeConfig::kFrameBuilder:
        SPDLOG_INFO("Building frame builder node.");
        root = FrameBuilder::Create(std::move(root), node_config.frame_builder());
        break;
      case pipeline::NodeConfig::kZmqSink:
        SPDLOG_INFO("Building ZMQ sink.");
        root = ZmqSink::Create(std::move(root), zmq_context, node_config.zmq_sink());
        break;
      case pipeline::NodeConfig::ROOT_NOT_SET:
        SPDLOG_WARN("No source type set. Ignoring node.");
        continue;
    }
  }

  return root;
}
