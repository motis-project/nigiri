#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

bool applicable(dir const&);
void load_timetable(source_idx_t, dir const&, timetable&);

}  // namespace nigiri::loader::gtfs