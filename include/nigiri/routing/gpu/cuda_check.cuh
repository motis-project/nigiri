#pragma once

#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

// Evaluates `code` once and throws on failure: an out-of-memory then fails
// the one search (as thrust's bad_alloc does) instead of the process. The
// runtime's last error is reset first, else a non-sticky error (an OOM) is
// reported again by the next unrelated check.
#define CUDA_CHECK(code)                                                     \
  do {                                                                       \
    if (auto const cuda_check_err = (code); cuda_check_err != cudaSuccess) { \
      cudaGetLastError();                                                    \
      throw std::runtime_error{std::string{"CUDA error: "} +                 \
                               cudaGetErrorString(cuda_check_err) + " at " + \
                               __FILE__ + ":" + std::to_string(__LINE__)};   \
    }                                                                        \
  } while (false)
