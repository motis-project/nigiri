#pragma once

#include <string_view>

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

using shape_id_map_t = hash_map<std::string, shape_idx_t>;

shape_id_map_t parse_shapes(std::string_view const, shape_vecvec_t&);

}  // namespace nigiri::loader::gtfs