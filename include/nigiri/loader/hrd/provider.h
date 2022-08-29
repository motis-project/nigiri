#pragma once

#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using provider_map_t = hash_map<string, provider_idx_t>;

provider_map_t parse_providers(config const&, timetable&, std::string_view);

}  // namespace nigiri::loader::hrd