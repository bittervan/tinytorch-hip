#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <stdexcept>

namespace py = pybind11;

// .hip 里实现
extern "C" void add_arrays_f32_host(const float* x, const float* y, float* out, std::size_t n);

// 简单的 NumPy 接口演示：把 host 数组搬到 device，调用 kernel，再搬回
py::array_t<float> add_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> x,
                             py::array_t<float, py::array::c_style | py::array::forcecast> y) {
  if (x.size() != y.size()) throw std::runtime_error("size mismatch");

  auto xreq = x.request(), yreq = y.request();
  if (xreq.ndim != 1 || yreq.ndim != 1) throw std::runtime_error("only 1D arrays for demo");

  const size_t n = static_cast<size_t>(x.size());
  float* hx = static_cast<float*>(xreq.ptr);
  float* hy = static_cast<float*>(yreq.ptr);

  // 输出
  py::array_t<float> out(n);
  auto oreq = out.request();
  float* ho = static_cast<float*>(oreq.ptr);

  add_arrays_f32_host(hx, hy, ho, n);

  // 恢复原 shape
  out.resize(xreq.shape);
  return out;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "Standalone HIP extension (demo)";
  m.def("add_numpy", &add_numpy, "elementwise add using HIP (numpy host arrays)");
}
