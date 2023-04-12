#pragma once

#include <algorithm>
#include <tuple>
#include <vector>

namespace nigiri {

template <typename T>
void apply_permutation(std::vector<unsigned> const& permutation,
                       T const& orig,
                       T& vec) {
  for (auto i = 0U; i != permutation.size(); ++i) {
    vec[i] = orig[permutation[i]];
  }
}

template <typename SortBy, typename... Ts>
std::tuple<std::decay_t<SortBy>, std::decay_t<Ts>...> sort_by(SortBy&& order,
                                                              Ts&&... ts) {
  std::vector<unsigned> permutation;
  permutation.resize(order.size());
  for (auto i = 0U; i != permutation.size(); ++i) {
    permutation[i] = i;
  }
  std::sort(begin(permutation), end(permutation),
            [&](auto&& a, auto&& b) { return order[a] < order[b]; });
  std::tuple<std::decay_t<SortBy>, std::decay_t<Ts>...> ret{order, ts...};
  std::apply(
      [&](SortBy& first, Ts&... rest) {
        apply_permutation(permutation, order, first);
        (apply_permutation(permutation, ts, rest), ...);
      },
      ret);
  return ret;
}

}  // namespace nigiri