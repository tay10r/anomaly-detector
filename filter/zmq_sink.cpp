#include "zmq_sink.h"

#include <spdlog/spdlog.h>
#include <zmq.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <vector>

#include "deps/stb_image_write.h"

namespace {

class ZmqSinkImpl final : public ZmqSink {
 public:
  ZmqSinkImpl(std::unique_ptr<Node> child_node, void* zmq_context, const pipeline::ZmqSinkConfig& config)
      : child_node_(std::move(child_node)), config_(config), socket_(zmq_socket(zmq_context, ZMQ_PUB)) {
    if (zmq_bind(socket_, config.bind_address().c_str()) != 0) {
      const auto err = errno;
      SPDLOG_ERROR("Failed to bind to '{}': {}", config.bind_address(), std::strerror(err));
    }
    int conflate{1};  // TODO : make it configurable
    zmq_setsockopt(&socket_, ZMQ_CONFLATE, &conflate, sizeof(conflate));
    SPDLOG_INFO("ZMQ sink publishing to '{}'.", config.bind_address());
  }

  ~ZmqSinkImpl()
  {
    zmq_close(socket_);
  }

  [[nodiscard]] auto Step() -> NodeOutput {
    auto child_output = child_node_->Step();
    if (child_output.EndOfStream()) {
      return NodeOutput();
    }

    std::vector<std::uint8_t> buffer;

    auto write_to_buffer = [](void* buffer_ptr, void* data, const int len) {
      auto* ptr = static_cast<std::vector<std::uint8_t>*>(buffer_ptr);
      const auto prev_size = ptr->size();
      ptr->resize(prev_size + len);
      std::memcpy(ptr->data() + prev_size, data, len);
    };

    const auto& img = *child_output.image;

    stbi_write_png_to_func(write_to_buffer, &buffer, img.Width(), img.Height(), 3, img.Data(), img.Width() * 3);

    zmq_msg_t msg{};

    zmq_msg_init_size(&msg, buffer.size());

    std::memcpy(zmq_msg_data(&msg), buffer.data(), buffer.size());  // TODO : use zero copy mechanism

    if (zmq_msg_send(&msg, socket_, 0) < 0) {
      const auto err = errno;
      SPDLOG_ERROR("Failed to send ZMQ message: {}", std::strerror(err));
    }

    zmq_msg_close(&msg);

    return child_output;
  }

 private:
  std::unique_ptr<Node> child_node_;

  pipeline::ZmqSinkConfig config_;

  void* socket_{};
};

}  // namespace

auto ZmqSink::Create(std::unique_ptr<Node> child_node, void* zmq_context,
                     const pipeline::ZmqSinkConfig& config) -> std::unique_ptr<ZmqSink> {
  return std::make_unique<ZmqSinkImpl>(std::move(child_node), zmq_context, config);
}
