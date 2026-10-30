#pragma once

#include "cista/gpu_compat.h"

namespace nigiri {

template <typename It, typename End, typename Key, typename Cmp>
CISTA_GPU_COMPAT It linear_lb(It from, End to, Key&& key, Cmp&& cmp) {
  for (auto it = from; it != to; ++it) {
    if (!cmp(*it, key)) {
      return it;
    }
  }
  return to;
}

}  // namespace nigiri
