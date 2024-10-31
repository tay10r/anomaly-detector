#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "deps/stb_image.h"
#include "deps/stb_image_write.h"

namespace {

namespace py = pybind11;

class Transform final {
 public:
  Transform(int seed) : rng_(seed) {}

  void SetNoiseLevel(std::uint8_t min_value, std::uint8_t max_value) {
    noise_min_ = min_value;
    noise_max_ = max_value;
  }

  void SetInfillSizes(const std::size_t x_min, const std::size_t y_min, const std::size_t x_max,
                      const std::size_t y_max) {
    x_min_ = x_min;
    y_min_ = y_min;
    x_max_ = x_max;
    y_max_ = y_max;
  }

  [[nodiscard]] auto Call(py::array_t<std::uint8_t, py::array::forcecast | py::array::c_style>& input)
      -> py::array_t<std::uint8_t> {
    const auto* shape = input.shape();

    if (input.ndim() != 3) {
      throw std::runtime_error("Only 3 axis tensors are supported.");
    }

    const auto num_channels = shape[0];
    const auto h = shape[1];
    const auto w = shape[2];

    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> out(
        std::vector<pybind11::ssize_t>{num_channels, h, w});

    std::uniform_int_distribution<std::size_t> x_dist(x_min_, x_max_);
    std::uniform_int_distribution<std::size_t> y_dist(y_min_, y_max_);

    const auto x_size = x_dist(rng_);
    const auto y_size = y_dist(rng_);

    if ((x_size > w) || (y_size > h)) {
      throw std::runtime_error("Region size is larger than image.");
    }

    std::uniform_int_distribution<std::size_t> x_offset_dist(0, w - x_size);
    std::uniform_int_distribution<std::size_t> y_offset_dist(0, h - y_size);

    const auto x_offset = x_offset_dist(rng_);
    const auto y_offset = y_offset_dist(rng_);

    const auto x_end = x_offset + x_size;
    const auto y_end = y_offset + y_size;

    std::uniform_int_distribution<int> noise_dist(noise_min_, noise_max_);

    for (auto c = 0; c < num_channels; c++) {
      auto* plane = out.mutable_data(c);

      const auto* in_ptr = input.data(c);

      for (auto y = 0; y < h; y++) {
        for (auto x = 0; x < w; x++) {
          const auto value = std::clamp(static_cast<int>(in_ptr[y * w + x]) + noise_dist(rng_), 0, 255);
          plane[y * w + x] = static_cast<std::uint8_t>(value);
        }
      }

      for (auto y = y_offset; y < y_end; y++) {
        for (auto x = x_offset; x < x_end; x++) {
          plane[y * w + x] = 0;
        }
      }
    }

    return out;
  }

 private:
  std::size_t x_min_{};
  std::size_t y_min_{};
  std::size_t x_max_{};
  std::size_t y_max_{};
  std::uint8_t noise_min_{};
  std::uint8_t noise_max_{};
  std::mt19937 rng_;
};

auto OpenImage(const char* path) -> py::array_t<std::uint8_t, py::array::c_style> {
  int w{};
  int h{};
  constexpr auto c{3};
  auto* pixels = stbi_load(path, &w, &h, nullptr, c);
  if (!pixels) {
    std::ostringstream stream;
    stream << "Failed to open '" << path << "'";
    throw std::runtime_error(stream.str());
  }

  py::array_t<std::uint8_t, py::array::c_style> output(std::vector<int>{c, h, w});

  auto* dst_r = output.mutable_data(0);
  auto* dst_g = output.mutable_data(1);
  auto* dst_b = output.mutable_data(2);

  for (int i = 0; i < c; i++) {
    for (int j = 0; j < h; j++) {
      for (auto k = 0; k < w; k++) {
        auto pixel = &pixels[(j * w + k) * c];
        dst_r[j * w + k] = pixel[0];
        dst_g[j * w + k] = pixel[1];
        dst_b[j * w + k] = pixel[2];
      }
    }
  }

  stbi_image_free(pixels);

  return output;
}

void SavePng(const char* path, const py::array_t<std::uint8_t, py::array::forcecast | py::array::c_style>& buffer) {
  if (buffer.ndim() != 3) {
    throw std::runtime_error("This function only supports 3 axis tensors.");
  }

  const auto c = buffer.shape()[0];
  const auto h = buffer.shape()[1];
  const auto w = buffer.shape()[2];

  std::vector<std::uint8_t> tmp(w * h * c);

  auto* src_r = buffer.data(0);
  auto* src_g = buffer.data(1);
  auto* src_b = buffer.data(2);

  for (auto i = 0; i < c; i++) {
    for (auto j = 0; j < h; j++) {
      for (auto k = 0; k < w; k++) {
        auto pixel = &tmp[(j * w + k) * c];
        pixel[0] = src_r[j * w + k];
        pixel[1] = src_g[j * w + k];
        pixel[2] = src_b[j * w + k];
      }
    }
  }

  stbi_write_png(path, w, h, c, tmp.data(), w * c);
}

}  // namespace

PYBIND11_MODULE(augmentation, m) {
  m.def("open_image", &OpenImage, py::arg("path"));

  m.def("save_png", &SavePng, py::arg("path"), py::arg("image"));

  py::class_<Transform>(m, "Transform")
      .def(py::init<int>(), py::arg("seed") = 0)
      .def("set_infill_sizes", &Transform::SetInfillSizes, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"),
           py::arg("y_max"))
      .def("set_noise_levels", &Transform::SetNoiseLevel, py::arg("min_value"), py::arg("max_value"))
      .def("__call__", &Transform::Call, py::arg("input"));
}
