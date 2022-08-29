#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using attribute_map_t = hash_map<string, attribute_idx_t>;

attribute_map_t parse_attributes(config const&,
                                 timetable&,
                                 utl::cstr const& file_content);

}  // namespace nigiri::loader::hrd
