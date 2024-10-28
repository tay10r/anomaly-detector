#include <spdlog/spdlog.h>
#include <zmq.h>

#include <cstdlib>
#include <cstring>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <string>

namespace {

struct Options final {
  int device_index{};

  std::string bind_address{"tcp://*:6020"};

  float interval{1.0F};

  int width{640};

  int height{480};

  bool help{false};

  void Parse(int argc, char** argv) {
    cxxopts::Options options(argv[0], "Publishes camera data over ZMQ.");
    options.add_options()  //
        ("d,device", "The video device to stream from.",
         cxxopts::value<int>()->default_value("0"))  //
        ("b,bind", "The address to bind the ZMQ publisher to.",
         cxxopts::value<std::string>()->default_value("tcp://*:6020"))  //
        ("w,width", "The width to capture frames at.",
         cxxopts::value<int>()->default_value("640"))  //
        ("h,height", "The height to capture frames at.",
         cxxopts::value<int>()->default_value("480"))  //
        ("t,interval", "The interval at which to publish camera frames",
         cxxopts::value<float>()->default_value("1.0"))  //
        ("help", "Prints this help message.",
         cxxopts::value<bool>()->implicit_value("true"))  //
        ;
    const auto result = options.parse(argc, argv);
    device_index = result["device"].as<int>();
    bind_address = result["bind"].as<std::string>();
    interval = result["interval"].as<float>();
    help = result["help"].as<bool>();
    if (help) {
      std::cout << options.help();
    }
  }
};

class Program final {
 public:
  Program()
      : zmq_context_(zmq_ctx_new()),
        zmq_publisher_(zmq_socket(zmq_context_, ZMQ_PUB)) {
    // The 'conflate' option keeps the outgoing queue to a size of one, and
    // ensures its the latest message.
    auto conflate{1};
    zmq_setsockopt(zmq_publisher_, ZMQ_CONFLATE, &conflate, sizeof(conflate));
  }

  ~Program() {
    zmq_close(zmq_publisher_);
    zmq_ctx_destroy(zmq_context_);
  }

  [[nodiscard]] auto Setup(int argc, char** argv) -> bool {
    try {
      options_.Parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
      SPDLOG_ERROR("{}", e.what());
      return false;
    }
    if (options_.help) {
      return false;
    }
    SPDLOG_INFO("Device Index: '{}'", options_.device_index);
    SPDLOG_INFO("Bind Address: '{}'", options_.bind_address);
    SPDLOG_INFO("Interval: {}", options_.interval);
    SPDLOG_INFO("Resolution: {}x{}", options_.width, options_.height);
    if (!m_video_device.open(options_.device_index)) {
      SPDLOG_ERROR("Failed to open video device {}", options_.device_index);
      return false;
    }
    m_video_device.set(cv::CAP_PROP_FRAME_WIDTH, options_.width);
    m_video_device.set(cv::CAP_PROP_FRAME_HEIGHT, options_.height);
    return true;
  }

  [[nodiscard]] auto NextFrame() -> bool {
    cv::Mat frame;

    if (!m_video_device.read(frame)) {
      SPDLOG_ERROR("Failed to read frame from video device.");
      return false;
    }

    std::vector<std::uint8_t> buffer;

    if (!cv::imencode(".png", frame, buffer)) {
      SPDLOG_ERROR("Failed to encode image.");
      return false;
    }

    zmq_msg_t msg{};

    zmq_msg_init_size(&msg, buffer.size());

    std::memcpy(zmq_msg_data(&msg), buffer.data(), buffer.size());

    if (zmq_msg_send(&msg, zmq_publisher_, 0) < 0) {
      SPDLOG_WARN("Failed to send message.");
    }

    zmq_msg_close(&msg);

    return true;
  }

 private:
  Options options_;

  void* zmq_context_{};

  void* zmq_publisher_{};

  cv::VideoCapture m_video_device;
};

}  // namespace

auto main(int argc, char** argv) -> int {
  Program program;
  if (!program.Setup(argc, argv)) {
    return EXIT_FAILURE;
  }
  while (program.NextFrame()) {
  }
  return EXIT_SUCCESS;
}
