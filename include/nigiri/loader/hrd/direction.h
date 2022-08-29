#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using direction_map_t = hash_map<string, string>;

direction_map_t parse_directions(config const&, std::string_view file_content);

}  // namespace nigiri::loader::hrd