#pragma once

#include <optional>

#include "utl/parser/cstr.h"

#include "cista/containers/vecvec.h"

#include "geo/latlng.h"
#include "geo/polyline.h"

#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

using shape_id_map_t = hash_map<std::string, shape_idx_t>;

shape_id_map_t const parse_shapes(std::string_view const,
                                  mm_vecvec<uint32_t, geo::latlng>*);

}  // namespace nigiri::loader::gtfs