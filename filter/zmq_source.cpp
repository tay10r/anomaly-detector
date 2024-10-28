#include "zmq_source.h"

#include <spdlog/spdlog.h>
#include <zmq.h>

#include <cerrno>
#include <cstring>

namespace {

class ZmqSourceImpl final : public ZmqSource {
 public:
  ZmqSourceImpl(void* zmq_context, const pipeline::ZmqSourceConfig& cfg) : socket_(zmq_socket(zmq_context, ZMQ_SUB)) {
    if (zmq_connect(socket_, cfg.connect_address().c_str()) != 0) {
      SPDLOG_ERROR("Failed to connect to '{}': {}", cfg.connect_address().c_str(), std::strerror(errno));
      failed_ = true;
    }
    zmq_setsockopt(socket_, ZMQ_SUBSCRIBE, "", 0);
    int conflate{1};
    zmq_setsockopt(socket_, ZMQ_CONFLATE, &conflate, 1);
  }

  [[nodiscard]] auto Step() -> NodeOutput override {
    if (failed_) {
      return NodeOutput();
    }

    auto img = std::make_shared<Image>();

    zmq_msg_t msg{};
    zmq_msg_init(&msg);
    if (zmq_msg_recv(&msg, socket_, 0) > 0) {
      if (!img->LoadFromMemory(zmq_msg_data(&msg), zmq_msg_size(&msg))) {
        SPDLOG_ERROR("Failed to load image from ZMQ subscriber.");
        return NodeOutput();
      }
      SPDLOG_INFO("Received image from ZMQ subscriber.");
    }
    zmq_msg_close(&msg);

    auto output = NodeOutput(std::move(img), frame_id_);
    frame_id_++;
    return output;
  }

 private:
  void* socket_{};

  bool failed_{};

  std::uint32_t frame_id_{};
};

}  // namespace

auto ZmqSource::Create(void* zmq_context, const pipeline::ZmqSourceConfig& cfg) -> std::unique_ptr<ZmqSource> {
  return std::make_unique<ZmqSourceImpl>(zmq_context, cfg);
}
