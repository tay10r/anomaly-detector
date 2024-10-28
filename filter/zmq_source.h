#pragma once

#include <pipeline/zmq_source_config.pb.h>

#include "node.h"

class ZmqSource : public Node {
 public:
  static auto Create(void* zmq_context, const pipeline::ZmqSourceConfig& config) -> std::unique_ptr<ZmqSource>;

  ~ZmqSource() = default;
};
