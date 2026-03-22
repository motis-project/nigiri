#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using direction_map_t = hash_map<string, translation_idx_t>;

direction_map_t parse_directions(config const&,
                                 timetable&,
                                 std::string_view file_content);

}  // namespace nigiri::loader::hrd