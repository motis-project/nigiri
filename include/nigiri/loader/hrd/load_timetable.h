#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::hrd {

bool applicable(config const&, dir const&);
std::uint64_t hash(config const&, dir const&, std::uint64_t seed = {});
void load_timetable(source_idx_t, config const&, dir const&, timetable&);

}  // namespace nigiri::loader::hrd
