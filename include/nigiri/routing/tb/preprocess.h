#pragma once

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

tb_data preprocess(timetable const&, profile_idx_t);

}  // namespace nigiri::routing::tb