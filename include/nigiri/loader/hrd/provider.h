#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/section_db.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using provider_map_t = hash_map<string, provider>;

provider_map_t parse_providers(config const&, std::string_view);

}  // namespace nigiri::loader::hrd