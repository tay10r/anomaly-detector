#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "deps/stb_image.h"
#include "deps/stb_image_write.h"

namespace {

namespace py = pybind11;

template <typename Scalar>
struct Rect final {
  Scalar x{};
  Scalar y{};
  Scalar w{};
  Scalar h{};
};

class Transform final {
 public:
  Transform(int seed) : rng_(seed) {}

  void SetNoiseRange(std::uint8_t min_value, std::uint8_t max_value) {
    noise_min_ = min_value;
    noise_max_ = max_value;
  }

  void SetInfillRect(const py::ssize_t x, const py::ssize_t y, const py::ssize_t w, const py::ssize_t h) {
    infill_rect_ = Rect<py::ssize_t>{x, y, w, h};
  }

  [[nodiscard]] auto Call(const py::array_t<std::uint8_t, py::array::forcecast | py::array::c_style>& input)
      -> py::tuple {
    const auto* shape = input.shape();

    if (input.ndim() != 3) {
      throw std::runtime_error("Only 3 axis tensors are supported.");
    }

    const auto num_channels = shape[0];
    const auto h = shape[1];
    const auto w = shape[2];

    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> target(
        std::vector<py::ssize_t>{num_channels, infill_rect_.w, infill_rect_.h});

    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> img(
        std::vector<py::ssize_t>{num_channels, h, w});

    std::uniform_int_distribution<int> noise_level_dist(noise_min_, noise_max_);
    const auto noise_level = noise_level_dist(rng_);
    std::uniform_int_distribution<int> noise_dist(-noise_level, noise_level);

    std::uniform_int_distribution<int> seed_dist(-1000, 1000);
    const auto call_seed{seed_dist(rng_)};
    for (auto c = 0; c < num_channels; c++) {
      auto* plane = img.mutable_data(c);

      const auto* in_ptr = input.data(c);

#pragma omp parallel for
      for (auto y = 0; y < h; y++) {
        std::seed_seq seed{y, c, call_seed};
        std::minstd_rand rng(seed);
        constexpr auto warmup{16};
        for (auto i = 0; i < warmup; i++) {
          (void)rng();
        }

        for (auto x = 0; x < w; x++) {
          const auto value = std::clamp(static_cast<int>(in_ptr[y * w + x]) + noise_dist(rng), 0, 255);
          plane[y * w + x] = static_cast<std::uint8_t>(value);
        }
      }

      auto* target_plane = target.mutable_data(c);
#pragma omp parallel for
      for (auto y = 0; y < infill_rect_.h; y++) {
        for (auto x = 0; x < infill_rect_.w; x++) {
          const auto value = in_ptr[(infill_rect_.y + y) * w + (infill_rect_.x + x)];
          target_plane[y * infill_rect_.w + x] = value;
          plane[(infill_rect_.y + y) * w + (infill_rect_.x + x)] = 0;
        }
      }
    }

    return py::make_tuple(img, target);
  }

  [[nodiscard]] auto Reconstruct(const py::array_t<std::uint8_t, py::array::forcecast | py::array::c_style>& input,
                                 const py::array_t<std::uint8_t, py::array::forcecast | py::array::c_style>& infill)
      -> py::array_t<std::uint8_t, py::array::forcecast | py::array::c_style> {
    if ((input.ndim() != 3) || (infill.ndim() != 3)) {
      throw std::runtime_error("Only 3 axis tensors are supported.");
    }

    if (infill.shape()[0] != input.shape()[0]) {
      throw std::runtime_error("The number of channels for the image and infill must be the same.");
    }

    if ((infill.shape()[2] != infill_rect_.w) || (infill.shape()[1] != infill_rect_.h)) {
      throw std::runtime_error("Infill size does not match transform settings.");
    }

    const auto num_channels = input.shape()[0];
    const auto h = input.shape()[1];
    const auto w = input.shape()[2];

    py::array_t<std::uint8_t, py::array::forcecast | py::array::c_style> output(
        std::vector<py::ssize_t>{num_channels, h, w});

    for (auto c = 0; c < num_channels; c++) {
      const auto* in_ptr = input.data(c);
      const auto* infill_ptr = infill.data(c);
      auto* out_ptr = output.mutable_data(c);
      std::memcpy(out_ptr, in_ptr, w * h);
      for (auto y = 0; y < infill_rect_.h; y++) {
        auto* out_row = out_ptr + (infill_rect_.y + y) * w + infill_rect_.x;
        const auto* infill_row = infill_ptr + y * infill_rect_.w;
        std::memcpy(out_row, infill_row, infill_rect_.w);
      }
    }

    return output;
  }

 private:
  Rect<py::ssize_t> infill_rect_{};

  std::uint8_t noise_min_{};

  std::uint8_t noise_max_{};

  std::minstd_rand rng_;
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
      .def("set_infill_rect", &Transform::SetInfillRect, py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"))
      .def("set_noise_range", &Transform::SetNoiseRange, py::arg("min_value"), py::arg("max_value"))
      .def("__call__", &Transform::Call, py::arg("input"))
      .def("reconstruct", &Transform::Reconstruct, py::arg("img"), py::arg("infill"));
}
