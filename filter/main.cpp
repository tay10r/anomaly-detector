#include <spdlog/spdlog.h>
#include <zmq.h>

#include <cstdlib>

#include "exception.h"
#include "node.h"

namespace {

class Program final {
 public:
  [[nodiscard]] auto Setup() -> bool {
    zmq_context_ = zmq_ctx_new();
    try {
      root_ = Node::CreatePipeline(zmq_context_, "pipeline.json");
    } catch (const Exception& e) {
      SPDLOG_ERROR("Failed to load pipeline: '{}'", e.what());
      return false;
    }
    return true;
  }

  void Teardown() {
    root_.reset();
    zmq_ctx_destroy(zmq_context_);
  }

  void Run() {
    while (true) {
      auto output = root_->Step();
      if (output.EndOfStream()) {
        SPDLOG_INFO("Reached end of stream.");
        break;
      }
    }
  }

 private:
  void* zmq_context_{};

  std::unique_ptr<Node> root_;
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
