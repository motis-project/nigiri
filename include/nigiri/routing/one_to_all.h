#pragma once

#include <limits>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct fastest_offset {
  delta_t duration_{std::numeric_limits<delta_t>::max()};
  std::uint8_t transfers_{std::numeric_limits<std::uint8_t>::max()};
};

template <direction SearchDir>
raptor_state one_to_all(timetable const& tt,
                        rt_timetable const* rtt,
                        query const& q);

template <direction SearchDir>
fastest_offset get_fastest_offset(timetable const& tt,
                                  raptor_state const& state,
                                  location_idx_t,
                                  unixtime_t start_time,
                                  std::uint8_t transfers);

}  // namespace nigiri::routing
