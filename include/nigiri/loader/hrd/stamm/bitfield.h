#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using bitfield_map_t = hash_map<unsigned, bitfield>;

bitfield hex_str_to_bitset(utl::cstr);

bitfield_map_t parse_bitfields(config const&, std::string_view file_content);

}  // namespace nigiri::loader::hrd
