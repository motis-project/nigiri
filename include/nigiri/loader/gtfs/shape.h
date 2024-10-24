#pragma once

#include <string_view>

#include "nigiri/types.h"

namespace nigiri {
struct shapes_storage;
}

namespace nigiri::loader::gtfs {

struct shape_state {
  shape_idx_t index_{};
  std::size_t last_seq_{};
};

struct shape_loader_state {
  hash_map<std::string, shape_state> id_map_{};
  paged_vecvec<shape_idx_t, double> distances_{};
  shape_idx_t index_offset_;
};

shape_loader_state parse_shapes(std::string_view const, shapes_storage&);

}  // namespace nigiri::loader::gtfs