#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::hrd {

void load_timetable(source_idx_t, config const&, dir const&, timetable&);

}  // namespace nigiri::loader::hrd
