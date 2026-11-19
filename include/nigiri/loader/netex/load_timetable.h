#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {
struct assistance_times;
}

namespace nigiri::loader::netex {

cista::hash_t hash(dir const& d);

bool applicable(dir const&);

void load_timetable(loader_config const&,
                    source_idx_t,
                    dir const&,
                    timetable&,
                    import_context&,
                    assistance_times* = nullptr,
                    shapes_storage* = nullptr);

}  // namespace nigiri::loader::netex
