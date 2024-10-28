#include "image.h"

#include <stb_image.h>

Image::~Image() {
  if (data_) {
    stbi_image_free(data_);
  }
}

auto Image::Load(const char* path) -> bool {
  int w{};
  int h{};
  auto* data = stbi_load(path, &w, &h, nullptr, 3);
  if (!data) {
    return false;
  }

  if (data_) {
    stbi_image_free(data_);
  }
  data_ = data;
  width_ = static_cast<uint32_t>(w);
  height_ = static_cast<uint32_t>(h);
  return true;
}
