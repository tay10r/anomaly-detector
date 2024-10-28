#include "source.h"

#include <google/protobuf/util/json_util.h>
#include <loader/config.pb.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <fstream>
#include <string>

#include "directory_source.h"
#include "exception.h"

namespace {

class NullSource final : public Source {
 public:
  auto GrabFrame() -> Image* override { return &null_image_; }

 private:
  Image null_image_{};
};

}  // namespace

auto Source::Create(const char* config_path) -> std::unique_ptr<Source> {
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

  loader::Config config;

  const auto status =
      google::protobuf::util::JsonStringToMessage(data, &config);
  if (!status.ok()) {
    throw Exception(std::string(status.message()));
  }

  switch (config.root_case()) {
    case loader::Config::kDirectory:
      return DirectorySource::Create(config.directory());
    case loader::Config::ROOT_NOT_SET:
      SPDLOG_WARN("No source type set, using null camera source.");
      return std::make_unique<NullSource>();
  }

  return nullptr;
}
