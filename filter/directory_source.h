#pragma once

#include <loader/directory_config.pb.h>

#include <memory>

#include "source.h"

class DirectorySource : public Source {
 public:
  static auto Create(const loader::DirectoryConfig& cfg)
      -> std::unique_ptr<DirectorySource>;

  ~DirectorySource() override = default;
};
