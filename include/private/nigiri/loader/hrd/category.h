#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/section_db.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

hash_map<std::string, info_db::handle_t> parse_categories(
    config const&, info_db&, std::string_view file_content);

}  // namespace nigiri::loader::hrd
