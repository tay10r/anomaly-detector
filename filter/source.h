#pragma once

#include <memory>

#include "image.h"

class Source {
 public:
  static auto Create(const char* config_path) -> std::unique_ptr<Source>;

  virtual ~Source() = default;

  [[nodiscard]] virtual auto GrabFrame() -> Image* = 0;
};
