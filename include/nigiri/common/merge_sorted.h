#pragma once

#include <cstddef>
#include <algorithm>

namespace nigiri {

// Merges src into the sorted dst, keeping each element once.
template <typename Dst, typename Src>
void merge_sorted(Dst& dst, Src const& src) {
  auto const n = static_cast<std::ptrdiff_t>(dst.size());
  dst.insert(end(dst), begin(src), end(src));
  std::sort(begin(dst) + n, end(dst));
  std::inplace_merge(begin(dst), begin(dst) + n, end(dst));
  dst.erase(std::unique(begin(dst), end(dst)), end(dst));
}

}  // namespace nigiri
