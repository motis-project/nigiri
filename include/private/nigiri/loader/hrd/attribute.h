#pragma once

#include "nigiri/loader/hrd/parse_config.h"
#include "nigiri/section_db.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

hash_map<std::string, info_db::handle_t> parse_attributes(
    config const&, info_db&, utl::cstr const& file_content);

}  // namespace nigiri::loader::hrd
