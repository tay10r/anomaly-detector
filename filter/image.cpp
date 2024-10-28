#include "image.h"

#include <stb_image.h>
#include <stb_image_write.h>

Image::Image(const uint32_t w, const uint32_t h) noexcept : data_(static_cast<uint8_t*>(std::malloc(w * h * 3))) {
  if (data_) {
    width_ = w;
    height_ = h;
  }
}

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

auto Image::LoadFromMemory(const void* data, const std::size_t size) -> bool {
  int w{};
  int h{};
  auto* ptr = stbi_load_from_memory(static_cast<const stbi_uc*>(data), size, &w, &h, nullptr, 3);
  if (!ptr) {
    return false;
  }

  if (data_) {
    stbi_image_free(data_);
  }
  data_ = ptr;
  width_ = static_cast<uint32_t>(w);
  height_ = static_cast<uint32_t>(h);
  return true;
}

auto Image::Save(const char* path) -> bool { return !!stbi_write_png(path, width_, height_, 3, data_, width_ * 3); }
