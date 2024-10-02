#pragma once

#include <ranges>
#include <string_view>
#include <type_traits>

#include "utl/helpers/algorithm.h"

#include "nigiri/shape.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct shape_state {
  shape_idx_t index_{};
  std::size_t last_seq_{};
};

struct shape_loader_state {
  hash_map<std::string, shape_state> id_map_{};
  // Stores median for each leg to allow small errors for each stop
  vecvec<shape_idx_t, double> distance_edges_{};
  shape_idx_t index_offset_;
};

shape_loader_state parse_shapes(std::string_view const, shapes_storage&);

template <typename DoubleRange>
  requires std::ranges::range<DoubleRange> &&
           std::is_same_v<std::ranges::range_value_t<DoubleRange>, double>
inline bool valid_distances(DoubleRange distances) {
  return utl::any_of(distances,
                     [](double const distance) { return distance > 0.0; });
}

}  // namespace nigiri::loader::gtfs