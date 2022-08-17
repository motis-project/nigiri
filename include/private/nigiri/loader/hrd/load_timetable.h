#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::hrd {

std::shared_ptr<timetable> load_timetable(config const&, dir const&);

}  // namespace nigiri::loader::hrd
