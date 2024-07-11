#pragma once

#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

void vdv_update(timetable const&,
                rt_timetable&,
                source_idx_t,
                std::string const& vdv_msg);

}  // namespace nigiri::rt