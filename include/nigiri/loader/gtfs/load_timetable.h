#pragma once

#include "geo/latlng.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {
struct assistance_times;
}

namespace nigiri::loader::gtfs {

cista::hash_t hash(dir const& d);

bool applicable(dir const&);

void load_timetable(loader_config const&,
                    source_idx_t,
                    dir const&,
                    timetable&,
                    assistance_times* = nullptr,
                    shape_vecvec_t* = nullptr);

void load_timetable(loader_config const&,
                    source_idx_t,
                    dir const&,
                    timetable&,
                    hash_map<bitfield, bitfield_idx_t>&,
                    assistance_times* = nullptr,
                    shape_vecvec_t* = nullptr);

}  // namespace nigiri::loader::gtfs