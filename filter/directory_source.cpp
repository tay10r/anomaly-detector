#include "directory_source.h"

#include <filesystem>

#include "image.h"

namespace {

class DirectorySourceImpl final : public DirectorySource {
 public:
  DirectorySourceImpl(const loader::DirectoryConfig& cfg) {
    const std::filesystem::path path{cfg.path().empty() ? std::string(".")
                                                        : cfg.path()};

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      Image img;
      if (img.Load(entry.path().string().c_str())) {
        images_.emplace_back(std::move(img));
      }
    }
  }

  auto GrabFrame() -> Image* override {
    if (images_.empty()) {
      return &null_image_;
    }
    auto& img = images_.at(offset_);
    offset_++;
    offset_ %= images_.size();
    return &img;
  }

 private:
  std::vector<Image> images_;

  std::size_t offset_{};

  Image null_image_;
};

}  // namespace

auto DirectorySource::Create(const loader::DirectoryConfig& cfg)
    -> std::unique_ptr<DirectorySource> {
  return std::make_unique<DirectorySourceImpl>(cfg);
}
