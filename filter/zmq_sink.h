#pragma once

#include <pipeline/zmq_sink_config.pb.h>

#include <memory>

#include "node.h"

class ZmqSink : public Node {
 public:
  static auto Create(std::unique_ptr<Node> child_node, void* zmq_context,
                     const pipeline::ZmqSinkConfig&) -> std::unique_ptr<ZmqSink>;

  ~ZmqSink() override = default;
};
