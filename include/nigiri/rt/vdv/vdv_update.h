#pragma once

#include <unordered_set>

#include "nigiri/rt/run.h"
#include "nigiri/rt/vdv/vdv_resolve_run.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

void vdv_update(timetable const&,
                rt_timetable&,
                source_idx_t const,
                std::string const& vdv_msg);

}  // namespace nigiri::rt