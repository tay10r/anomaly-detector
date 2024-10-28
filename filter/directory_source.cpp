#include "directory_source.h"

#include <algorithm>
#include <filesystem>
#include <limits>

#include "image.h"

namespace {

class DirectorySourceImpl final : public DirectorySource {
 public:
  DirectorySourceImpl(const pipeline::DirectorySourceConfig& cfg) {
    const std::filesystem::path path{cfg.path().empty() ? std::string(".")
                                                        : cfg.path()};

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      paths_.emplace_back(entry.path().string());
    }

    std::sort(paths_.begin(), paths_.end());
  }

  auto Step() -> NodeOutput override {
    auto img = std::make_shared<Image>();

    auto id{std::numeric_limits<std::uint32_t>::max()};

    for (auto i = offset_; i < paths_.size(); i++) {
      if (img->Load(paths_[offset_].c_str())) {
        offset_ = (i + 1);
        id = i;
        break;
      }
    }

    // Note: If all images fail to load or there are no images in the directory,
    // we may end up here with an empty image.

    return NodeOutput(img, id);
  }

 private:
  std::vector<std::string> paths_;

  std::size_t offset_{};
};

}  // namespace

auto DirectorySource::Create(const pipeline::DirectorySourceConfig& cfg)
    -> std::unique_ptr<DirectorySource> {
  return std::make_unique<DirectorySourceImpl>(cfg);
}
