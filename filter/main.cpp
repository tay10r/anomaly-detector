#include <spdlog/spdlog.h>
#include <zmq.h>

#include <cstdlib>

#include "exception.h"
#include "source.h"

namespace {

class Program final {
 public:
  [[nodiscard]] auto Setup() -> bool {
    zmq_context_ = zmq_ctx_new();
    socket_ = zmq_socket(zmq_context_, ZMQ_REQ);
    try {
      source_ = Source::Create("loader.json");
    } catch (const Exception& e) {
      SPDLOG_ERROR("Failed to load source: '{}'", e.what());
      return false;
    }
    return true;
  }

  void Teardown() {
    zmq_close(socket_);
    source_.reset();
    zmq_ctx_destroy(zmq_context_);
  }

  void Run() {
    while (true) {
      ReadNextMessage();
    }
  }

 protected:
  void HandleRequest() {
    auto frame = source_->GrabFrame();

    //
  }

  void ReadNextMessage() {
    zmq_msg_t msg{};

    zmq_msg_init(&msg);

    if (zmq_msg_recv(&msg, socket_, 0) >= 0) {
      HandleRequest();
    }

    zmq_msg_close(&msg);
  }

 private:
  void* zmq_context_{};

  void* socket_{};

  std::unique_ptr<Source> source_;
};

}  // namespace

auto main() -> int {
  Program program;
  if (!program.Setup()) {
    return EXIT_FAILURE;
  }
  program.Run();
  program.Teardown();
  return EXIT_SUCCESS;
}
