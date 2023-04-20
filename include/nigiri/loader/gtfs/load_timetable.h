#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

cista::hash_t hash(dir const& d);
bool applicable(dir const&);
void load_timetable(source_idx_t, dir const&, timetable&);

}  // namespace nigiri::loader::gtfs