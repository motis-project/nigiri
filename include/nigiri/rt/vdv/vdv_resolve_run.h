#pragma once

#include <optional>
#include <string>

#include "date/date.h"

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::rt {

struct vdv_run;

constexpr auto const kAllowedError = minutes_after_midnight_t::rep{5};

std::optional<location_idx_t> match_location(timetable const&,
                                             std::string_view vdv_stop_id);

template <event_type ET>
void match_time(timetable const&,
                location_idx_t,
                unixtime_t,
                hash_set<transport>& matches);

hash_set<transport> match_transport(timetable const&, vdv_run const&);

}  // namespace nigiri::rt