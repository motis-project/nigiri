#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/types.h"

#include "td_timetable.h"

namespace nigiri::loader {
struct assistance_times;
}

namespace nigiri::loader::gtfs_flex {

cista::hash_t hash(dir const& d);

bool applicable(dir const&);

td_timetable_map_t load_td_timetable(loader_config const& config,
                    source_idx_t src,
                    dir const& d,
                    assistance_times* assistance = nullptr);

td_timetable_map_t load_td_timetable(loader_config const& config,
                    source_idx_t src,
                    dir const& d,
                    hash_map<bitfield, bitfield_idx_t>& h,
                    assistance_times* assistance = nullptr);

}  // namespace nigiri::loader::gtfs_flex