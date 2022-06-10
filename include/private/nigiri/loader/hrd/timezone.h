#pragma once

#include <chrono>
#include <string_view>

#include "nigiri/loader/hrd/eva_number.h"
#include "nigiri/loader/hrd/parser_config.h"
#include "nigiri/section_db.h"
#include "nigiri/types.h"

namespace nigiri::loader::hrd {

using timezone_map_t = std::map<eva_number, tz_offsets>;

tz_offsets const& get_tz(timezone_map_t const&, eva_number const);

inline bool is_local_time_in_season(
    tz_offsets const& tz,
    unixtime_t const day,
    duration_t const local_minutes_after_midnight) {
  using std::chrono::sys_days;
  auto const local_time = day + local_minutes_after_midnight;
  return tz.season_.has_value() &&
         tz.season_->begin_ + tz.offset_ <= local_time &&
         local_time <= tz.season_->end_ + tz.season_->offset_;
}

inline std::pair<duration_t, int> local_mam_to_utc_mam(
    tz_offsets const& tz,
    unixtime_t const day,
    duration_t const local_mam,
    int initial_day_shift = 0) {
  auto const is_season = is_local_time_in_season(tz, day, local_mam);
  auto const offset = is_season ? tz.season_->offset_ : tz.offset_;
  auto const utc_mam =
      initial_day_shift * duration_t{1440} + local_mam + offset;
  if (utc_mam < duration_t{0}) {
    return {utc_mam + duration_t{1440}, -1};
  } else {
    return {utc_mam, 0};
  }
}

timezone_map_t parse_timezones(config const&, std::string_view);

}  // namespace nigiri::loader::hrd
