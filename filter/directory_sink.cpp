#include "directory_sink.h"

#include <stb_image_write.h>

#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace {

class DirectorySinkImpl final : public DirectorySink {
 public:
  DirectorySinkImpl(std::unique_ptr<Node> child, const pipeline::DirectorySinkConfig& cfg)
      : child_(std::move(child)), config_(cfg) {}

  auto Step() -> NodeOutput override {
    auto output = child_->Step();
    if (output.EndOfStream()) {
      return output;
    }
    std::ostringstream name_stream;
    name_stream << std::setw(8) << std::setfill('0') << image_index_ << ".png";
    auto path{std::filesystem::path(config_.path()) / name_stream.str()};

    (void)output.image->Save(path.string().c_str());

    image_index_++;

    return output;
  }

 private:
  std::unique_ptr<Node> child_;

  pipeline::DirectorySinkConfig config_;

  std::size_t image_index_{};
};

}  // namespace

auto DirectorySink::Create(std::unique_ptr<Node> child,
                           const pipeline::DirectorySinkConfig& cfg) -> std::unique_ptr<DirectorySink> {
  return std::make_unique<DirectorySinkImpl>(std::move(child), cfg);
}
