#pragma once

#include <string_view>
#include <vector>

#include "nigiri/shape.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct shape_state {
  shape_idx_t index_{};
  std::size_t last_seq_{};
  double last_distance_traveled_;
};

struct shape_loader_state {
  hash_map<std::string, shape_state> id_map_{};
  vector_map<shape_idx_t, std::vector<double>> distances_{};
  shape_idx_t index_offset_;
};

shape_loader_state parse_shapes(std::string_view const, shapes_storage&);

}  // namespace nigiri::loader::gtfs