#pragma once

#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing::tb {

enum class parallelization : std::uint8_t { kParallel, kSequential };

tb_data preprocess(timetable const&,
                   profile_idx_t,
                   parallelization = parallelization::kSequential);

}  // namespace nigiri::routing::tb