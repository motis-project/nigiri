#pragma once

#include "nigiri/routing/limits.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct footpath;
}  // namespace nigiri

namespace nigiri::routing {
struct query;

template <direction SearchDir>
void lb_raptor(
    timetable const&,
    query const&,
    bitvec& station_mark,
    bitvec& prev_station_mark,
    bitvec& is_start,
    vector_map<location_idx_t, std::array<std::uint16_t, kMaxTransfers + 2U>>&
        location_round_lb);

}  // namespace nigiri::routing