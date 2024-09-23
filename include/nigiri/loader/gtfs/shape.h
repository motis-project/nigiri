#pragma once

#include <string_view>
#include <vector>

#include "nigiri/shape.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

struct shape_state {
  shape_idx_t index_{};
  std::size_t last_seq_{};
  std::vector<double> distances_{};
};

using shape_id_map_t = hash_map<std::string, shape_state>;

shape_id_map_t parse_shapes(std::string_view const, shapes_storage&);

}  // namespace nigiri::loader::gtfs