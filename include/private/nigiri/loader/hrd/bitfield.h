#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/section_db.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using bitfield_map_t =
    hash_map<unsigned, std::pair<bitfield, info_db::handle_t>>;

bitfield hex_str_to_bitset(utl::cstr hex, int line_number);

bitfield_map_t parse_bitfields(config const&, info_db&,
                               std::string_view file_content);

}  // namespace nigiri::loader::hrd
