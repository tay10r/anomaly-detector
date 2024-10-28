#pragma once

#include <pipeline/directory_sink_config.pb.h>

#include <memory>

#include "node.h"

class DirectorySink : public Node {
 public:
  static auto Create(std::unique_ptr<Node> child,
                     const pipeline::DirectorySinkConfig& cfg)
      -> std::unique_ptr<DirectorySink>;

  ~DirectorySink() override = default;
};
