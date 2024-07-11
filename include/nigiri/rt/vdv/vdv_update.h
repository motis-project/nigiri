#pragma once

#include <unordered_set>

#include "nigiri/rt/run.h"
#include "nigiri/rt/vdv/vdv_run.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

constexpr auto const kAllowedError = minutes_after_midnight_t::rep{5};

std::optional<location_idx_t> match_location(timetable const&,
                                             std::string_view vdv_stop_id);

template <event_type ET>
void match_time(timetable const&,
                location_idx_t const,
                unixtime_t const,
                hash_set<transport>& matches);

hash_set<transport> match_transport(timetable const&, vdv_run const&);

void vdv_update(timetable const&,
                rt_timetable&,
                source_idx_t const,
                std::string const& vdv_msg);

}  // namespace nigiri::rt