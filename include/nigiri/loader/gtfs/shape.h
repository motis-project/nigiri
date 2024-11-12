#pragma once

#include <string_view>

#include "nigiri/types.h"

namespace nigiri {
struct shapes_storage;
}

namespace nigiri::loader::gtfs {

using relative_shape_idx_t =
    cista::strong<std::uint32_t, struct relative_shape_idx_>;

struct shape_loader_state {
  shape_idx_t get_shape_idx(relative_shape_idx_t const i) const {
    return shape_idx_t{to_idx(i) + index_offset_};
  }
  relative_shape_idx_t get_relative_idx(shape_idx_t const i) const {
    return relative_shape_idx_t{to_idx(i) - index_offset_};
  }

  hash_map<std::string, shape_idx_t> id_map_{};
  vector_map<relative_shape_idx_t, std::vector<double>> distances_{};
  shape_idx_t::value_t index_offset_;
};

shape_loader_state parse_shapes(std::string_view const, shapes_storage&);

}  // namespace nigiri::loader::gtfs