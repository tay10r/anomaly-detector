#pragma once

#include <stdint.h>

class Image final {
 public:
  Image() noexcept = default;

  Image(uint32_t w, uint32_t h) noexcept;

  Image(const uint32_t w, const uint32_t h, uint8_t* data) noexcept : width_(w), height_(h), data_(data) {}

  Image(Image&& other) noexcept : width_(other.width_), height_(other.height_), data_(other.data_) {
    other.width_ = 0;
    other.height_ = 0;
    other.data_ = nullptr;
  }

  Image(const Image&) = delete;

  ~Image();

  auto operator=(const Image&) -> Image& = delete;

  auto operator=(Image&&) -> Image& = delete;

  [[nodiscard]] auto Load(const char* path) -> bool;

  [[nodiscard]] auto Save(const char* path) -> bool;

  [[nodiscard]] auto Data() -> uint8_t* { return data_; }

  [[nodiscard]] auto Data() const -> const uint8_t* { return data_; }

  [[nodiscard]] auto Width() const -> uint32_t { return width_; }

  [[nodiscard]] auto Height() const -> uint32_t { return height_; }

  [[nodiscard]] auto Empty() const -> bool { return (width_ == 0) || (height_ == 0); }

 private:
  uint32_t width_{};

  uint32_t height_{};

  uint8_t* data_{};
};
