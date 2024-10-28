#pragma once

#include <pipeline/directory_source_config.pb.h>

#include <memory>

#include "node.h"

class DirectorySource : public Node {
 public:
  static auto Create(const pipeline::DirectorySourceConfig& cfg)
      -> std::unique_ptr<DirectorySource>;

  ~DirectorySource() override = default;
};
